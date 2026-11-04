"""Optional Metal compensated products, with explicit CPU comparison policy.

Metal has no native float64 here. Each real operand is represented by two
float32 words; FMA product errors and compensated sums are retained. This is
not IEEE float64 and is never silently substituted for a qualification operator.
"""
import numpy as np


_SOURCE = r'''
#include <metal_stdlib>
using namespace metal;
inline float2 twosum(float a, float b) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    float s=a+b; float z=s-a;
    return float2(s, (a-(s-z))+(b-z));
}
inline float2 add_words(float2 a, float2 b) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    float2 s=twosum(a.x,b.x);
    return twosum(s.x, (s.y+a.y)+b.y);
}
inline float2 multiply_words(float2 x, float2 y) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    float p=x.x*y.x;
    float error=fma(x.x,y.x,-p);
    error=error+x.x*y.y; error=error+x.y*y.x; error=error+x.y*y.y;
    return twosum(p,error);
}
kernel void compensated_product(device const float2* a, device const float2* b,
    device float2* out, constant uint& m, constant uint& n, constant uint& k,
    uint gid [[thread_position_in_grid]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    if (gid>=m*n) return;
    uint i=gid/n, j=gid%n; float2 sum=float2(0.0f,0.0f);
    for(uint l=0;l<k;l++) {
        float2 x=a[i*k+l], y=b[l*n+j];
        sum=add_words(sum,multiply_words(x,y));
    }
    out[gid]=sum;
}
kernel void residual_product(device const float2* residual, device const float2* values,
    device float2* out, constant uint& columns, constant uint& size,
    uint gid [[thread_position_in_grid]]) {
    if(gid<size) out[gid]=multiply_words(residual[gid/columns],values[gid]);
}
'''


class MetalCompensatedProducts:
    """Explicit optional backend; full checking is the safe qualification mode.

    Sample checking is a diagnostic only, not an error bound for unsampled
    entries. Eigensolvers using it must qualify with an independent CPU operator.
    Importing this module does not import torch or initialize a GPU.
    """
    approximate_products = True

    def __init__(self, *, comparison='full', absolute_tolerance=1e-10,
            relative_tolerance=5e-13, maximum_workspace_bytes=128*1024**3):
        if (comparison not in ('full', 'sampled') or not np.isfinite(absolute_tolerance)
                or absolute_tolerance <= 0 or not np.isfinite(relative_tolerance)
                or relative_tolerance <= 0 or maximum_workspace_bytes < 1):
            raise ValueError('explicit comparison policy and positive tolerances/resources required')
        import torch
        if not torch.backends.mps.is_available():
            raise RuntimeError('Metal/MPS is unavailable; no implicit CPU fallback')
        self.torch = torch; self.library = torch.mps.compile_shader(_SOURCE)
        self.comparison = comparison
        self.atol, self.rtol = absolute_tolerance, relative_tolerance
        self.maximum_workspace_bytes = maximum_workspace_bytes
        self._tensor_cache = None
        self.diagnostics = dict(arithmetic='two-word float32 compensated FMA',
            native_float64=False, comparison=comparison, products=0,
            maximum_checked_error=0., comparison_is_global_bound=False)

    def _words(self, values):
        high = values.astype(np.float32)
        low = (values-high.astype(np.float64)).astype(np.float32)
        words = np.ascontiguousarray(np.stack((high, low), axis=-1))
        return self.torch.from_numpy(words).to('mps')

    def matmul(self, left, right):
        a, b = np.asarray(left), np.asarray(right)
        if (a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]
                or np.iscomplexobj(a) or np.iscomplexobj(b)
                or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b))):
            raise ValueError('compatible finite real matrices required')
        m, k = a.shape; n = b.shape[1]
        if not m or not k or not n or m*n > np.iinfo(np.uint32).max:
            raise ValueError('nonempty Metal uint32 grid required')
        required = 64*(a.size+b.size+m*n)
        if required > self.maximum_workspace_bytes:
            raise MemoryError('Metal compensated product exceeds workspace budget')
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        if max(float(abs(a).max()), float(abs(b).max())) > 1e15:
            raise ValueError('operands exceed qualified compensated product range')
        av, bv = self._words(a), self._words(b)
        out = self.torch.empty((m, n, 2), dtype=self.torch.float32, device='mps')
        self.library.compensated_product(av, bv, out, m, n, k, threads=m*n)
        words = out.cpu().numpy()
        result = words[..., 0].astype(np.float64)+words[..., 1].astype(np.float64)
        rows = np.arange(m) if self.comparison == 'full' else np.unique(np.linspace(0, m-1, min(m, 8), dtype=int))
        cols = np.arange(n) if self.comparison == 'full' else np.unique(np.linspace(0, n-1, min(n, 8), dtype=int))
        reference = a[rows]@b[:, cols]
        scale = abs(a[rows])@abs(b[:, cols])
        error = abs(result[np.ix_(rows, cols)]-reference)
        if not np.all(np.isfinite(result)) or np.any(error > self.atol+self.rtol*scale):
            raise ArithmeticError('Metal compensated product failed its CPU comparison')
        self.diagnostics['products'] += 1
        self.diagnostics['maximum_checked_error'] = max(
            self.diagnostics['maximum_checked_error'], float(error.max()))
        return result

    def tensor_quadrature_action(self, functions, residual, basis, packets):
        """Keep compensated words on Metal through every modal transform.

        Only initial packets and final retained coefficients cross the device
        boundary. Frozen modal tables, residual and retained-row map are cached.
        Full CPU comparison, when selected, is performed by the tensor operator.
        """
        from math import prod
        x = np.asarray(packets)
        if np.iscomplexobj(x) or not np.all(np.isfinite(x)):
            raise ValueError('compensated Metal tensor action requires finite real packets')
        modal_shape = tuple(len(f) for f in functions)
        grid_shape = tuple(f.shape[1] for f in functions)
        largest = prod(max(m, q) for m, q in zip(modal_shape, grid_shape))
        if 64*largest*x.shape[1]+32*residual.size > self.maximum_workspace_bytes:
            raise MemoryError('persistent Metal tensor action exceeds workspace budget')
        key = (tuple(id(f) for f in functions), id(residual), id(basis))
        cacheable = not any(v.flags.writeable for v in (*functions, residual, basis))
        if self._tensor_cache is None or self._tensor_cache['key'] != key or not cacheable:
            tables = [(self._words(np.ascontiguousarray(f)), self._words(np.ascontiguousarray(f.T)))
                      for f in functions]
            positions = np.ravel_multi_index(tuple(basis.T), modal_shape)
            cache = dict(key=key, owners=(functions, residual, basis), tables=tables,
                residual=self._words(residual.reshape(-1, 1)),
                positions=self.torch.from_numpy(positions.copy()).to('mps'))
            self._tensor_cache = cache if cacheable else None
        else:
            cache = self._tensor_cache
        coefficients = np.zeros(modal_shape+(x.shape[1],), dtype=np.float64)
        coefficients[tuple(basis.T)] = x
        values = self._words(coefficients)
        def transform(value, axis, table, output_rows):
            moved = value.movedim(axis, 0).contiguous()
            shape = tuple(moved.shape); inner = shape[0]
            columns = prod(shape[1:-1])
            right = moved.reshape(inner, columns, 2)
            out = self.torch.empty((output_rows, columns, 2), dtype=self.torch.float32, device='mps')
            self.library.compensated_product(table, right, out, output_rows, columns, inner,
                                             threads=output_rows*columns)
            self.diagnostics['products'] += 1
            return out.reshape((output_rows,)+shape[1:]).movedim(0, axis)
        for axis, (_, transpose) in enumerate(cache['tables']):
            values = transform(values, axis, transpose, grid_shape[axis])
        values = values.contiguous(); weighted = self.torch.empty_like(values)
        self.library.residual_product(cache['residual'], values, weighted, x.shape[1],
            prod(grid_shape)*x.shape[1], threads=prod(grid_shape)*x.shape[1])
        values = weighted
        for axis in reversed(range(len(functions))):
            values = transform(values, axis, cache['tables'][axis][0], modal_shape[axis])
        retained = values.reshape(prod(modal_shape), x.shape[1], 2).index_select(0, cache['positions'])
        words = retained.cpu().numpy()
        result = words[..., 0].astype(np.float64)+words[..., 1].astype(np.float64)
        if not np.all(np.isfinite(result)):
            raise ArithmeticError('nonfinite compensated Metal tensor action')
        self.diagnostics['persistent_tensor_action'] = True
        return result
