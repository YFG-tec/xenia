"""Plot the tstop=5e5 inner-user / outer-outflow dump (no re-run)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "output" / "tend5e5"
R_P = 1.36 * 7.1492e9
R_HILL = 4.5
FLUIDS = ("HI", "HEI", "HII", "HEII")
STYLES = {"HI": "-", "HEI": "--", "HII": "-.", "HEII": ":"}


def main() -> None:
    fin = np.load(OUT / "final.npz")
    hist = np.load(OUT / "hist.npz")
    r = fin["r"]
    rr = r / R_P
    t = hist["t"]

    n_e = fin["HII_n"] + fin["HEII_n"]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4))
    for s in FLUIDS:
        ax[0].plot(rr, fin[f"{s}_n"], ls=STYLES[s], label=s)
        ax[1].plot(rr, fin[f"{s}_vx"], ls=STYLES[s], label=s)
        ax[2].plot(rr, fin[f"{s}_T"], ls=STYLES[s], label=s)
    ax[0].plot(rr, n_e, color="0.15", lw=2.0, label=r"$n_e=n_{\mathrm{HII}}+n_{\mathrm{HEII}}$")
    ax[2].plot(rr, fin["T_e"], color="0.15", lw=2.0, label=r"$T_e$")
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"$r / R_p$")
    ax[0].set_ylabel(r"$n$, cm$^{-3}$")
    ax[0].legend(fontsize=7, ncol=2)
    ax[0].set_title("number densities (+ electrons)")
    ax[1].axhline(0.0, color="0.5", lw=0.8)
    ax[1].set_xlabel(r"$r / R_p$")
    ax[1].set_ylabel(r"$v_r$, cm/s")
    ax[1].legend(fontsize=8)
    ax[1].set_title("radial velocities (no $v_e$ fluid)")
    ax[2].set_xlabel(r"$r / R_p$")
    ax[2].set_ylabel(r"$T$, K")
    ax[2].legend(fontsize=7)
    ax[2].set_title("temperatures")
    fig.tight_layout()
    fig.savefig(OUT / "profiles.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(rr, fin["HII_P"], label=r"$P_{\mathrm{HII}}$")
    ax[0].plot(rr, fin["HEII_P"], ls="--", label=r"$P_{\mathrm{HEII}}$")
    ax[0].plot(rr, fin["P_e"], color="0.15", lw=2.0, label=r"$P_e$")
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"$r / R_p$")
    ax[0].set_ylabel(r"$P$, erg/cm$^3$")
    ax[0].legend(fontsize=8)
    ax[0].set_title("ion + electron pressures")
    ax[1].plot(rr, fin["HII_n"] / np.maximum(fin["HI_n"], 1e-300),
               label=r"HII/HI")
    ax[1].plot(rr, fin["HEII_n"] / np.maximum(fin["HEI_n"], 1e-300),
               ls="--", label=r"HEII/HEI")
    ax[1].plot(rr, n_e / np.maximum(fin["HI_n"] + fin["HEI_n"], 1e-300),
               color="0.15", lw=2.0, label=r"$n_e/(n_{\mathrm{HI}}+n_{\mathrm{HEI}})$")
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$r / R_p$")
    ax[1].set_ylabel("ion fraction")
    ax[1].legend(fontsize=8)
    ax[1].set_title("ion fractions")
    fig.tight_layout()
    fig.savefig(OUT / "pressure_ionfrac.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4))
    ax[0].plot(rr, n_e, color="C0", lw=1.8, label=r"$n_e$")
    ax[0].plot(rr, fin["HII_n"], ls="--", label=r"$n_{\mathrm{HII}}$")
    ax[0].plot(rr, fin["HEII_n"], ls=":", label=r"$n_{\mathrm{HEII}}$")
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"$r / R_p$")
    ax[0].set_ylabel(r"$n$, cm$^{-3}$")
    ax[0].legend(fontsize=8)
    ax[0].set_title(r"$n_e = n_{\mathrm{HII}} + n_{\mathrm{HEII}}$")
    ax[1].plot(rr, fin["T_e"], color="C0", lw=1.8, label=r"$T_e$")
    ax[1].plot(rr, fin["HII_T"], ls="--", label=r"$T_{\mathrm{HII}}$")
    ax[1].set_xlabel(r"$r / R_p$")
    ax[1].set_ylabel(r"$T$, K")
    ax[1].legend(fontsize=8)
    ax[1].set_title("electron vs HII temperature")
    ax[2].plot(rr, fin["P_e"], color="C0", lw=1.8, label=r"$P_e$")
    ax[2].set_yscale("log")
    ax[2].set_xlabel(r"$r / R_p$")
    ax[2].set_ylabel(r"$P_e$, erg/cm$^3$")
    ax[2].legend(fontsize=8)
    ax[2].set_title("electron pressure")
    fig.tight_layout()
    fig.savefig(OUT / "electrons.png", dpi=140)
    plt.close(fig)

    md = fin["mdot"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(rr, md, lw=1.6)
    ax[0].axvline(R_HILL, color="0.5", ls=":", lw=0.8)
    ax[0].axhline(0.0, color="0.5", lw=0.8)
    ax[0].set_yscale("symlog", linthresh=1e8)
    ax[0].set_xlabel(r"$r / R_p$")
    ax[0].set_ylabel(r"$\dot{M}$, g/s")
    ax[0].set_title(r"$\dot{M}(r)$")
    mask = rr >= R_HILL
    ax[1].plot(rr[mask], md[mask], lw=1.6)
    ax[1].axhline(0.0, color="0.5", lw=0.8)
    ax[1].set_xlabel(r"$r / R_p$")
    ax[1].set_ylabel(r"$\dot{M}$, g/s")
    ax[1].set_title(r"outflow $r > 4.5 R_p$")
    fig.tight_layout()
    fig.savefig(OUT / "mass_flux.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(14, 3.5))
    ax[0].plot(t, hist["dt"])
    ax[0].set_xlabel("t, s")
    ax[0].set_ylabel("dt, s")
    ax[1].plot(t, hist["resid"])
    ax[1].set_yscale("log")
    ax[1].set_xlabel("t, s")
    ax[1].set_ylabel("source_resid")
    ax[2].plot(t, hist["mdot_top"], label="top")
    ax[2].plot(t, hist["mdot_mid"], ls="--", label="mid")
    ax[2].set_xlabel("t, s")
    ax[2].set_ylabel(r"$\dot{M}$, g/s")
    ax[2].legend(fontsize=7)
    ax[3].plot(t, hist["ion_h_top"], label="H top")
    ax[3].plot(t, hist["ion_he_top"], ls="--", label="He top")
    ax[3].set_yscale("log")
    ax[3].set_xlabel("t, s")
    ax[3].set_ylabel("ion fraction")
    ax[3].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "history.png", dpi=140)
    plt.close(fig)
    print("wrote", OUT / "profiles.png")
    print("wrote", OUT / "pressure_ionfrac.png")
    print("wrote", OUT / "electrons.png")
    print("wrote", OUT / "mass_flux.png")
    print("wrote", OUT / "history.png")


if __name__ == "__main__":
    main()
