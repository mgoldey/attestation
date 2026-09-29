# Catalyst screen and basis-set check (illustrative)

A worked example for `attest`: the runs and the numbers are invented for the
demo, not measured. Every number below carries a `claim:` annotation naming
the run it came from, so `attest claims` can re-derive it from the files in
`results/` rather than trusting the prose.

Two of these claims are deliberately wrong, and one cites a key no reference
list holds. Finding them is the demo.

## Catalyst screen

Each coupling ran at 80 °C for 12 h, in triplicate, at 2.5 mol% catalyst loading.

Palladium acetate gave the best isolated yield, 78%.
<!-- claim: catalysis/screen_pd metric=yield value=78 cite=smith2031palladium -->

The nickel catalyst reached 68%, close enough to justify a scale-up.
<!-- claim: catalysis/screen_ni metric=yield value=68 -->

Copper iodide was clearly worse, at 41%.
<!-- claim: catalysis/screen_cu metric=yield value=41 -->

Iron chloride, the cheapest option, gave 55%.
<!-- claim: catalysis/screen_fe metric=yield value=55 -->

## Basis-set convergence

Against the benchmark subset, cc-pVTZ brings the mean absolute error to
1.42 kcal/mol, and cc-pVQZ to 1.37 kcal/mol at roughly ten times the cost.
<!-- claim: dft/basis_tz metric=mae value=1.42 cite=goerigk2017look -->
<!-- claim: dft/basis_qz metric=mae value=1.37 cite=dunning1989gaussian -->
