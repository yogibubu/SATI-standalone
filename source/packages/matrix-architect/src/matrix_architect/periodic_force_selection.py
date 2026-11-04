"""Select periodic E/G representations using grouped training deletions only."""
import numpy as np
from .fourier_force_fit import fit_regularized_fourier_energy_gradient


def fourier_values(field,points):
    phase=np.asarray(points)@np.asarray(field.wavevectors).T
    return field.constant_hartree+np.cos(phase)@field.cosine_coefficients_hartree+np.sin(phase)@field.sine_coefficients_hartree


def select_periodic_force_fields(points,energies,gradients,candidates,*,anchor_index,
        shape_points,minimum_energy,energy_scale,gradient_scale,retention_factor=1.25):
    """Return training-selected fields and an explicitly scored representation set.

    Each candidate is a keyword dictionary for the existing regularized fitter.
    Each non-reference geometry is deleted together with all its derivatives.
    Shape inequalities constrain interpolation geometries without electronic
    evaluations. Eligibility also requires the stated full-training E/G gates.
    Returned fields are frozen before any external controls are inspected.
    """
    q=np.asarray(points,float);e=np.asarray(energies,float);g=np.asarray(gradients,float)
    if (len(q)<3 or not np.isfinite(energy_scale) or energy_scale<=0
            or not np.isfinite(gradient_scale) or gradient_scale<=0
            or not np.isfinite(retention_factor) or retention_factor<1):
        raise ValueError('At least three training geometries and finite positive scales required')
    rows=[];fields={}
    for index,candidate in enumerate(candidates):
        row=dict(index=index,parameters=candidate)
        kwargs=dict(candidate,lower_bound_points=shape_points,lower_bound_values=minimum_energy)
        try:
            field,audit=fit_regularized_fourier_energy_gradient(q,e,g,anchor_index=anchor_index,**kwargs)
            errors=[]
            for omit in range(len(q)):
                if omit==anchor_index:continue
                keep=[i for i in range(len(q)) if i!=omit]
                fit,_=fit_regularized_fourier_energy_gradient(q[keep],e[keep],g[keep],anchor_index=keep.index(anchor_index),**kwargs)
                value,gradient,_=fit.derivatives(q[omit]);errors.append([value-e[omit],*(gradient-g[omit])])
            errors=np.array(errors);full=np.array([[v[0]-energy,*(v[1]-gradient)] for v,energy,gradient in zip([field.derivatives(x) for x in q],e,g)])
            energy=float(abs(full[:,0]).max());gradient=float(abs(full[:,1:]).max())
            score=float(max(np.sqrt(np.mean(errors[:,0]**2))/energy_scale,np.sqrt(np.mean(errors[:,1:]**2))/gradient_scale))
            row.update(deletion_score=score,deletion_errors=errors.tolist(),training_errors=full.tolist(),
                maximum_training_energy_error=energy,maximum_training_gradient_error=gradient,
                eligible=energy<=energy_scale and gradient<=gradient_scale,audit=audit)
            if row['eligible']:fields[index]=field
        except (ValueError,np.linalg.LinAlgError) as error:row.update(eligible=False,error=str(error))
        rows.append(row)
    eligible=[r for r in rows if r['eligible']]
    if not eligible:return dict(candidates=rows,selected_index=None,retained_indices=[]),{}
    selected=min(eligible,key=lambda r:(r['deletion_score'],r['index']))
    retained=[r['index'] for r in eligible if r['deletion_score']<=selected['deletion_score']*retention_factor]
    return dict(candidates=rows,selected_index=selected['index'],retained_indices=retained,
        retention_factor=retention_factor,selection_uses_external_controls=False),{i:fields[i] for i in retained}
