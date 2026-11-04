# LINK/SMITH rapido per GA e Monte Carlo intermolecolari

La nuova API non sostituisce le API LINK/SMITH esistenti: i programmi attuali
continuano a funzionare. Per complessi rigidi, una popolazione può ora essere
convertita direttamente da 6 coordinate per partner a cartesiane, senza
costruire la matrice di Wilson B e senza iterare l'exponential mapping.

Lo stesso principio si applica alle torsioni intramolecolari a geometria
rigida. Quando le sole coordinate spostate sono dihedrals aciclici
one-primitive e il legame centrale separa il grafo molecolare, LINK applica
un'unica rotazione axis--angle finita al sottografo mobile. Il percorso
`DIRECT_RIGID_ACYCLIC_TORSION` valuta soltanto i valori delle coordinate e non
costruisce la matrice B. Tutte le coordinate congelate vengono poi verificate:
se una rotazione è accoppiata, appartiene a un anello o altera un'altra
coordinata, LINK torna automaticamente al back-transform SONIC ibrido
predictor--corrector.

Torsioni e pose intermolecolari possono ora cambiare anche nella stessa
struttura. LINK compila una sola volta i ponti torsionali e i frame rigidi,
applica tutte le rotazioni finite usando i valori correnti già archiviati,
quindi impone analiticamente le pose rispetto al frame di riferimento
eventualmente deformato. Il percorso `DIRECT_RIGID_SOFT_COORDINATES` esegue
una sola verifica value-only finale; non costruisce B e non entra nel
back-transform completo. Il fallback resta automatico se una coordinata non è
un diedro aciclico rigido o una posa `FTRANS/FROT` completa.

Il fast path comprende anche il puckering endociclico, da solo oppure insieme
a torsioni rigide acicliche e pose intermolecolari. I blocchi RPCK/U/D sono
compilati una sola volta dalla definizione SONIC congelata; a runtime LINK
costruisce soltanto le righe B locali dell'anello e nella line search valuta
soltanto i valori locali. I percorsi
`DIRECT_RIGID_RING_PUCKERING` e `DIRECT_RIGID_RING_SOFT_COORDINATES`
concludono con una verifica value-only di tutte le coordinate. Se un anello
fuso, un accoppiamento o una coordinata stiff non restano invariati, si torna
automaticamente al predictor--corrector SONIC completo. La selezione del
percorso non cambia la topologia durante l'esplorazione: deriva sempre dal
contratto continuo congelato.

## Requisiti

La definizione SMITH deve essere congelata in C1 e contenere, per ogni partner
mobile, le terne complete `FTRANS(1:3)` e `FROT(1:3)` rispetto allo stesso
frammento di riferimento. Coordinate intramolecolari o vincoli generici restano
supportati dal normale fallback LINK.

Dopo aver aggiornato il checkout MATRIX:

```bash
cd /percorso/del/MATRIX
source scripts/matrix_env.sh
matrix-set
python -m pip install -e packages/matrix-link
```

L'ultimo comando aggiorna anche l'entry point del servizio. Nell'ambiente
MATRIX si può sempre usare l'equivalente
`python -m matrix_link.pose_service`.

## Uso Python consigliato

```python
from matrix_link import RigidComplexModel
from matrix_smith import read_gic_definition_from_xyzin

definition = read_gic_definition_from_xyzin("complesso.xyzin")
model = RigidComplexModel.from_definition(definition)

# population: array (n_candidati, n_coordinate_SONIC)
cartesiane = model.realize_sonic_batch(population, workers=8)
# risultato: (n_candidati, n_atomi, 3), in angstrom
```

Per un algoritmo che conserva traslazioni e quaternioni:

```python
poses0 = model.reference_poses()
poses1 = model.mutate_pose(
    poses0,
    fragment_index=0,
    translation_increment_angstrom=(0.1, 0.0, -0.1),
    rotation_increment_radian=(0.0, 0.2, 0.0),
)
cartesiane = model.realize_batch((poses0, poses1), workers=8)
```

Conviene conservare l'orientazione come quaternione e usare il vettore
esponenziale solo come incremento locale della mutazione.

## Servizio persistente per un programma esterno

Avviare una sola volta:

```bash
matrix-link-pose-service complesso.xyzin --workers 8
```

Il processo legge una richiesta JSON per riga da `stdin` e scrive una risposta
per riga su `stdout`. Non va rilanciato per ogni candidato.

```json
{"id":"pop-1","op":"realize_sonic_batch","values":[[0,0,3,0,0,0],[0.1,0,3.1,0,0.2,0]]}
```

La risposta contiene `coordinates_angstrom`. L'ordine dei candidati è
preservato anche con più worker. `{"op":"describe"}` restituisce dimensioni e
indici. Ogni richiesta batch può sovrascrivere il default con `"workers":4`.

Per descrittori non coincidenti con le sei coordinate di posa usare
`realize_constraints_batch`, passando `coordinate_indices` e una matrice
`target_values`: vengono calcolate soltanto le righe SMITH richieste e i
candidati sono risolti in parallelo.

## Compatibilità e scelta dei worker

- Il codice corrente basato su `GeometryEvaluationService` non richiede
  modifiche e usa automaticamente il fast path per pose rigide, diedri
  aciclici rigidi, puckering endociclico e loro combinazioni.
- `coordinates_from_q_batch` usa il kernel NumPy vettoriale per popolazioni
  pose-only; il backend residente riceve direttamente questo batch e non
  rivaluta `actual_q` dopo una trasformazione diretta certificata.
- Nei casi generali si può impostare
  `OptimizerSettings(coordinate_parallel_workers=8)` per parallelizzare le
  righe B selezionate.
- Per scansioni Monte Carlo usare `PESExplorationPolicy.monte_carlo()`, che
  evita simmetrizzazioni punto per punto non necessarie.
- Inviare batch abbastanza grandi (indicativamente almeno 100 candidati per
  worker). Se il GA parallelizza già calcoli quantistici esterni, evitare
  parallelismo annidato e lasciare `workers=1` nel servizio.

Il protocollo completo e i dettagli matematici sono in
[`../../docs/manuals/link_rigid_pose_service.md`](../../docs/manuals/link_rigid_pose_service.md).
