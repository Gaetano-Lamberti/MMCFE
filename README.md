# Mathematical Modeling for Chemical and Food Engineering — companion repository

Companion repository of the textbook

> G. Lamberti, *Mathematical Modeling for Chemical and Food Engineering*, 1st edition (in preparation).

Every script printed or discussed in the book is here, in **Python** and in **MATLAB**.
The two versions of each script have the same structure and give the same results.
The only exceptions are `PDE_CaseA_CylindricalTransient.m` and `PDE_Example_CanSterilization_solvepde.m` (Chapter 5), which exist in MATLAB only.

## Contents

The repository has two folders:

- `Lectures/` – slides of the course lessons (PDF). Lesson *n* corresponds to Chapter *n* of the book.
- `Scripts/` – the scripts, one subfolder per chapter:

| Folder | Book chapter | Scripts |
|---|---|---|
| `Scripts/ch03_Algebraic_Equations` | 3 – Algebraic Equations | Newton–Raphson and variants; Examples 3.1–3.2 (CSTR) |
| `Scripts/ch04_Ordinary_Differential_Equations` | 4 – Ordinary Differential Equations | Examples 1–4; built-in solvers; BVP; DAE; Examples 4.1–4.2 (PFR) |
| `Scripts/ch05_Partial_Differential_Equations` | 5 – Partial Differential Equations | Cases A–C; can sterilization; slab drying; laminar tubular reactor |
| `Scripts/ch06_Parameter_Estimation` | 6 – Parameter Estimation | Thermistor, thin-layer drying, Herschel–Bulkley; batch-reactor labs (data generation, fitting, Arrhenius, confidence intervals) |
| `Scripts/ch07_Constrained_Optimization` | 7 – Constrained Optimization | Tableau simplex method; bakery product mix, least-cost burger, thermal process design |

Further chapters will be added as the book is completed.
The problems solved in the case histories are stated in Chapter 2 (the guiding example: catalytic
dehydrogenation of ethylbenzene to styrene).

## How to use

**Without git:** click the green **Code** button → **Download ZIP**, then unzip.

**Python** (3.9 or later):

```
pip install -r requirements.txt
python Scripts/ch03_Algebraic_Equations/Example_3_1_CSTR_isothermal.py
```

**MATLAB** (R2016b or later): open the `.m` file and press **Run** (F5). Most scripts need no toolbox; the exceptions are `PDE_CaseB_Laplace2D.m` and `PDE_Example_CanSterilization_solvepde.m` (Partial Differential Equation Toolbox) and `Lab5_BatchFitting_LM_k1k2.m`, `Lab7_GlobalFit_4Params.m` (Optimization Toolbox).

## Errata and questions

Errors in the book or in the code are listed in [`ERRATA.md`](ERRATA.md).
To report a new one, open an [issue](../../issues).

## How to cite

Use the **Cite this repository** button (top right), or the DOI badge of the Zenodo archive.

## License

Code: [MIT](LICENSE). Data and figures: CC BY 4.0.
