"""Functions that are useful for WEC analysis and design.
"""


from __future__ import annotations


__all__ = [
    "plot_hydrodynamic_coefficients",
    "plot_bode_impedance",
    "calculate_power_flows",
    "plot_power_flow",
    "colors",
]


from typing import Optional, Union
import logging
from pathlib import Path

import autograd.numpy as np
from autograd.numpy import ndarray

from xarray import DataArray
from numpy.typing import ArrayLike
# from autograd.numpy import ndarray
from xarray import DataArray, concat
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from matplotlib.sankey import Sankey
from wecopttool import WEC
from wecopttool.pto import PTO

# logger
_log = logging.getLogger(__name__)


def power_flow_colors():
    clrs = {'hydro':        (0.267004, 0.004874, 0.329415, 1.0), #viridis(0.0)
        'hydro_mech':   (0.229739, 0.322361, 0.545706, 1.0), #viridis(0.25)
        'mech':         (0.127568, 0.566949, 0.550556, 1.0), #viridis(0.5)
        'mech_elec':    (0.369214, 0.788888, 0.382914, 1.0), #viridis(0.75)
        'elec':         (0.974417, 0.90359, 0.130215, 0.5), #viridis(0.99)
        }
    return clrs

def plot_hydrodynamic_coefficients(bem_data,
                                   wave_dir: Optional[float] = 0.0
                                   )-> list(tuple(Figure, Axes)):
    """Plots hydrodynamic coefficients (added mass, radiation damping,
       and wave excitation) based on BEM data.

    Parameters
    ----------
    bem_data
        Linear hydrodynamic coefficients obtained using the boundary
        element method (BEM) code Capytaine, with sign convention
        corrected.
    wave_dir
        Wave direction(s) to plot.
    """

    bem_data = bem_data.sel(wave_direction = wave_dir, method='nearest')
    radiating_dofs = bem_data.radiating_dof.values
    influenced_dofs = bem_data.influenced_dof.values

    # plots
    fig_am, ax_am = plt.subplots(
        len(radiating_dofs), 
        len(influenced_dofs),
        tight_layout=True, 
        sharex=True, 
        figsize=(3*len(radiating_dofs),3*len(influenced_dofs)),
        squeeze=False
        )
    fig_rd, ax_rd = plt.subplots(
        len(radiating_dofs),
        len(influenced_dofs),
        tight_layout=True,
        sharex=True, 
        figsize=(3*len(radiating_dofs), 3*len(influenced_dofs)),
        squeeze=False
        )
    fig_ex, ax_ex = plt.subplots(
        len(influenced_dofs),
        1,
        tight_layout=True, 
        sharex=True, 
        figsize=(3, 3*len(radiating_dofs)), 
        squeeze=False
        )
    [ax.grid(True) for axs in (ax_am, ax_rd, ax_ex) for ax in axs.flatten()]
    # plot titles
    fig_am.suptitle('Added Mass Coefficients', fontweight='bold')
    fig_rd.suptitle('Radiation Damping Coefficients', fontweight='bold')
    fig_ex.suptitle('Wave Excitation Coefficients', fontweight='bold')

    sp_idx = 0
    for i, rdof in enumerate(radiating_dofs):
        for j, idof in enumerate(influenced_dofs):
            sp_idx += 1
            if i == 0:
                np.abs(bem_data.diffraction_force.sel(influenced_dof=idof)).plot(
                    ax=ax_ex[j,0], linestyle='dashed', label='Diffraction')
                np.abs(bem_data.Froude_Krylov_force.sel(influenced_dof=idof)).plot(
                    ax=ax_ex[j,0], linestyle='dashdot', label='Froude-Krylov')
                ex_handles, ex_labels = ax_ex[j,0].get_legend_handles_labels()
                ax_ex[j,0].set_title(f'{idof}')
                ax_ex[j,0].set_xlabel('')
                ax_ex[j,0].set_ylabel('')
            if j <= i:
                bem_data.added_mass.sel(
                    radiating_dof=rdof, influenced_dof=idof).plot(ax=ax_am[i, j])
                bem_data.radiation_damping.sel(
                    radiating_dof=rdof, influenced_dof=idof).plot(ax=ax_rd[i, j])
                if i == len(radiating_dofs)-1:
                    ax_am[i, j].set_xlabel(f'$\omega$', fontsize=10)
                    ax_rd[i, j].set_xlabel(f'$\omega$', fontsize=10)
                    ax_ex[j, 0].set_xlabel(f'$\omega$', fontsize=10)
                else:
                    ax_am[i, j].set_xlabel('')
                    ax_rd[i, j].set_xlabel('')
                if j == 0:
                    ax_am[i, j].set_ylabel(f'{rdof}', fontsize=10)
                    ax_rd[i, j].set_ylabel(f'{rdof}', fontsize=10)
                else:
                    ax_am[i, j].set_ylabel('')
                    ax_rd[i, j].set_ylabel('')
                if j == i:
                    ax_am[i, j].set_title(f'{idof}', fontsize=10)
                    ax_rd[i, j].set_title(f'{idof}', fontsize=10)
                else:
                    ax_am[i, j].set_title('')
                    ax_rd[i, j].set_title('')
            else:
                fig_am.delaxes(ax_am[i, j])
                fig_rd.delaxes(ax_rd[i, j])
    fig_ex.legend(ex_handles, ex_labels, loc=(0.08, 0), ncol=2, frameon=False)
    return [(fig_am,ax_am), (fig_rd,ax_rd), (fig_ex,ax_ex)]

def plot_bode_impedance(impedance: DataArray, 
                        title: Optional[str]= '',
                        fig_axes: Optional[list(Figure, Axes)] = None,
                        #plot_natural_freq: Optional[bool] = False,
)-> tuple(Figure, Axes):
    """Plot Bode graph from wecoptool impedance data array.

    Parameters
    ----------
    impedance
        Complex impedance matrix produced by for example by
        :py:func:`wecopttool.hydrodynamic_impedance`.
        Dimensions: omega, radiating_dofs, influenced_dofs
    title
        Title string to be displayed in the plot.
    """
    radiating_dofs = impedance.radiating_dof.values
    influenced_dofs = impedance.influenced_dof.values
    mag = 20.0 * np.log10(np.abs(impedance))
    phase = np.rad2deg(np.unwrap(np.angle(impedance)))
    freq = impedance.omega.values/2/np.pi   
    if fig_axes is None:
        fig, axes = plt.subplots(
            2*len(radiating_dofs), 
            len(influenced_dofs),
            tight_layout=True, 
            sharex=True, 
            figsize=(3*len(radiating_dofs), 3*len(influenced_dofs)), 
            squeeze=False
            )
    else:
        fig = fig_axes[0]
        axes = fig_axes[1]
    fig.suptitle(title + ' Bode Plots', fontweight='bold')

    sp_idx = 0
    for i, rdof in enumerate(radiating_dofs):
        for j, idof in enumerate(influenced_dofs):
            sp_idx += 1
            axes[2*i, j].semilogx(freq, mag[:, i, j])    # Bode magnitude plot
            axes[2*i+1, j].semilogx(freq, phase[:, i, j])    # Bode phase plot
            axes[2*i, j].grid(True, which = 'both')
            axes[2*i+1, j].grid(True, which = 'both')
            if i == len(radiating_dofs)-1:
                axes[2*i+1, j].set_xlabel(f'Frequency (Hz)', fontsize=10)
            else:
                axes[i, j].set_xlabel('')
            if j == 0:
                axes[2*i, j].set_ylabel(f'{rdof} \n Mag. (dB)', fontsize=10)
                axes[2*i+1, j].set_ylabel(f'Phase. (deg)', fontsize=10)
            else:
                axes[i, j].set_ylabel('')
            if i == 0:
                axes[i, j].set_title(f'{idof}', fontsize=10)
            else:
                axes[i, j].set_title('')
    return fig, axes


def calculate_power_flows(
    wec: WEC, 
    pto: PTO, 
    results: OptimizeResult, 
    waves: Dataset, 
    intrinsic_impedance: DataArray
) -> dict[str, float]:
    """Calculate power flows into a :py:class:`wecopttool.WEC`
    and through a :py:class:`wecopttool.pto.PTO` based on the results
    of :py:meth:`wecopttool.WEC.solve` for a single wave realization.

    This function returns a dictionary containing the power flows, which can
    be used as input for the :py:func:`plot_power_flow` function.

    Parameters
    ----------
    wec : WEC
        WEC object of :py:class:`wecopttool.WEC`.
    
    pto : PTO
        PTO object of :py:class:`wecopttool.pto.PTO`.
    
    results : OptimizeResult
        Results produced by :py:func:`scipy.optimize.minimize` for a single wave
        realization.
    
    waves : Dataset
        An :py:class:`xarray.Dataset` with the structure and elements
        shown by :py:mod:`wecopttool.waves`.
    
    intrinsic_impedance : DataArray
        Complex intrinsic impedance matrix produced by 
        :py:func:`wecopttool.hydrodynamic_impedance`.
        Dimensions: omega, radiating_dofs, influenced_dofs.

    Returns
    -------
    dict[str, float]
        A dictionary containing the calculated power flows, with keys such as
        'Optimal Excitation', 'Deficit Excitation', 'Excitation', 
        'Deficit Radiated', 'Radiated', 'Absorbed', 
        'Electrical', 'Mechanical', and 'PTO Loss'.
    """
    wec_fdom, _ = wec.post_process(wec, results, waves)
    x_wec, x_opt = wec.decompose_state(results[0].x)

    #power quntities from solver
    P_mech = pto.mechanical_average_power(wec, x_wec, x_opt, waves)
    P_elec = pto.average_power(wec, x_wec, x_opt, waves)

    #compute analytical power flows
    Fexc_FD_full = wec_fdom[0].force.sel(type=
                        ['Froude_Krylov',
                         'diffraction']).sum('type')
    Rad_res = np.real(intrinsic_impedance.squeeze())
    Vel_FD = wec_fdom[0].vel

    P_max_abs, P_exc, P_rad = [], [], []

    #This solution requires radiation resistance matrix Rad_res to be invertible
    # TODO In the future we might want to add an entirely unconstrained solve 
    # for optimized mechanical power

    for om in Rad_res.omega.values:   
        #use frequency vector from intrinsic impedance (no zero freq)
        #Eq. 6.69
        #Dofs are row vector, which is transposed in standard convention
        Fexc_FD_t = np.atleast_2d(Fexc_FD_full.sel(omega = om))    
        Fexc_FD = np.transpose(Fexc_FD_t)
        R_inv = np.linalg.inv(np.atleast_2d(Rad_res.sel(omega= om)))
        P_max_abs.append((1/8)*(Fexc_FD_t@R_inv)@np.conj(Fexc_FD)) 
        #Eq.6.57
        U_FD_t = np.atleast_2d(Vel_FD.sel(omega = om))
        U_FD = np.transpose(U_FD_t)
        R = np.atleast_2d(Rad_res.sel(omega= om))
        P_rad.append((1/2)*(U_FD_t@R)@np.conj(U_FD))
        #Eq. 6.56 (replaced pinv(Fe)*U with U'*conj(Fe) 
        # as suggested in subsequent paragraph)
        P_exc.append((1/4)*(Fexc_FD_t@np.conj(U_FD) + U_FD_t@np.conj(Fexc_FD)))

    power_flows = {
        'Optimal Excitation' : 2* np.sum(np.real(P_max_abs)),#eq 6.68 
        'Max Absorbed': 1* np.sum(np.real(P_max_abs)),
        'Radiated': 1*np.sum(np.real(P_rad)), 
        'Excitation': 1*np.sum(np.real(P_exc)), 
        'Electrical': -1*P_elec, 
        'Mechanical': -1*P_mech, 
                  }

    power_flows['Absorbed'] =  (
        power_flows['Excitation'] 
        - power_flows['Radiated']
            )
    power_flows['Deficit Excitation'] =  (
        power_flows['Optimal Excitation'] 
        - power_flows['Excitation']
            )
    power_flows['Deficit Absorbed'] =  (
        power_flows['Max Absorbed'] 
        - power_flows['Absorbed']
            ) 
    power_flows['Deficit Radiated'] =  (
        power_flows['Deficit Excitation'] 
        - power_flows['Deficit Absorbed']
            )     
    power_flows['PTO Loss'] = (
        power_flows['Mechanical'] 
        -  power_flows['Electrical']
            )
    return power_flows


def plot_power_flow(power_flows: dict[str, float], 
                    plot_reference: bool = True,
                    axes_title: str = '', 
                    axes: Axes = None,
                    return_fig_and_axes: bool = False
    )-> tuple(Figure, Axes):
    """Plot power flow through a WEC as a Sankey diagram.
   
   If you are not considering a model with mechanical and
   electrical components, you will need to customize this function.

    Parameters
    ----------
    power_flows : dict[str, float]
        A dictionary containing power flow values produced by, for example,
        :py:func:`wecopttool.utilities.calculate_power_flows`.
        Required keys include:
            - 'Optimal Excitation'
            - 'Deficit Excitation'
            - 'Excitation'
            - 'Deficit Radiated'
            - 'Radiated'
            - 'Absorbed'
            - 'Electrical'
            - 'Mechanical'
            - 'PTO Loss'
    
    plot_reference : bool, optional
        If True, the reference power will be plotted. Default is True.
    
    axes_title : str, optional
        A string to display as the title over the Sankey diagram. Default is an empty string.
    
    axes : Axes, optional
        A Matplotlib Axes object where the Sankey diagram will be drawn. If None, a new figure and axes will be created. Default is None.
    
    return_fig_and_axes : bool, optional
        If True, the function will return the Figure and Axes objects. Default is False.

    Returns
    -------
    tuple[Figure, Axes]
        A tuple containing the Matplotlib Figure and Axes objects.
    """

    if axes is None:
        fig, axes = plt.subplots(nrows = 1, ncols= 1,
                tight_layout=True, 
                figsize= [8, 4])
    clrs = power_flow_colors()
    len_trunk = 1.0
    if plot_reference:
        sankey = Sankey(ax=axes, 
                        scale= 1/power_flows['Optimal Excitation'],
                        offset= 0,
                        format = '%.1f',
                        shoulder = 0.02,
                        tolerance=1e-03*power_flows['Optimal Excitation'],
                        unit = 'W')
        sankey.add(flows=[power_flows['Optimal Excitation'],
                    -1*power_flows['Deficit Excitation'],
                    -1*power_flows['Excitation']], 
            labels = [' Optimal \n Excication ', 
                    'Deficit \n Excitation', 
                    'Excitation'], 
            orientations=[0, 0,  0],#arrow directions,
            pathlengths = [0.15,0.15,0.15],
            trunklength = len_trunk,
            edgecolor = 'None',
            facecolor = clrs['hydro'],
                alpha = 0.1,
            label = 'Reference',
                )
        n_diagrams = 1
        init_diag  = 0
        if power_flows['Deficit Excitation'] > 0.1:
            sankey.add(flows=[power_flows['Deficit Excitation'],
                        -1*power_flows['Deficit Radiated'],
                        -1*power_flows['Deficit Absorbed'],], 
                labels = ['XX Deficit Exc', 
                        'Deficit \n Radiated',
                            'Deficit \n Absorbed', ], 
                prior= (0),
                connect=(1,0),
                orientations=[0, 1,  0],#arrow directions,
                pathlengths = [0.15,0.01,0.15],
                trunklength = len_trunk,
                edgecolor = 'None',
                facecolor = clrs['hydro_mech'],
                alpha = 0.3, #viridis(0.2)
                label = 'Reference',
                    )
            n_diagrams = n_diagrams +1
    else:
        sankey = Sankey(ax=axes, 
                        scale= 1/power_flows['Excitation'],
                        offset= 0,
                        format = '%.1f',
                        shoulder = 0.02,
                        tolerance=1e-03*power_flows['Excitation'],
                        unit = 'W')
        n_diagrams = 0
        init_diag = None

    sankey.add(flows=[power_flows['Excitation'],
                        -1*(power_flows['Absorbed'] 
                           + power_flows['Radiated'])], 
                labels = ['Excitation', 
                        'Excitaion'], 
                prior = init_diag,
                connect=(2,0),
                orientations=[0,  -0],#arrow directions,
                pathlengths = [.15,0.15],
                trunklength = len_trunk,
                edgecolor = 'None',
                facecolor = clrs['hydro'] #viridis(0.9)
        )
    sankey.add(flows=[
                (power_flows['Absorbed'] + power_flows['Radiated']),
                -1*power_flows['Radiated'],
                -1*power_flows['Absorbed']], 
                labels = ['Excitation', 
                        'Radiated', 
                        'Absorbed'], 
                # prior= (0),
                prior= (n_diagrams),
                connect=(1,0),
                orientations=[0, -1,  -0],#arrow directions,
                pathlengths = [0.15,0.2,0.15],
                trunklength = len_trunk-0.2,
                edgecolor = 'None', 
                facecolor = clrs['hydro_mech'] #viridis(0.5)
        )
    sankey.add(flows=[power_flows['Absorbed'],
                        -1*power_flows['Mechanical']], 
                labels = ['Absorbed', 
                        'Mechanical'], 
                prior= (n_diagrams+1),
                connect=(2,0),
                orientations=[0,  -0],#arrow directions,
                pathlengths = [.15,0.15],
                trunklength = len_trunk,
                edgecolor = 'None',
                facecolor = clrs['mech'] #viridis(0.9)
        )
    sankey.add(flows=[(power_flows['Mechanical']),
                        -1*power_flows['PTO Loss'],
                        -1*power_flows['Electrical']], 
                labels = ['Mechanical', 
                        'PTO-Loss' , 
                        'Electrical'], 
                prior= (n_diagrams+2),
                connect=(1,0),
                orientations=[0, -1,  -0],#arrow directions,
                pathlengths = [.15,0.2,0.15],
                trunklength = len_trunk,
                edgecolor = 'None',
                facecolor = clrs['mech_elec'] #viridis(0.9)
        )
    sankey.add(flows=[(power_flows['Electrical']),
                        -1*power_flows['Electrical']], 
                labels = ['', 
                        'Electrical'], 
                prior= (n_diagrams+3),
                connect=(2,0),
                orientations=[0,  -0],#arrow directions,
                pathlengths = [.15,0.15],
                trunklength = len_trunk,
                edgecolor = 'None',
                facecolor = clrs['elec'] #viridis(0.9)
        )


    diagrams = sankey.finish()

    for diagram in diagrams:
        for text in diagram.texts:
            text.set_fontsize(8)

    #Remvove labels that are double
    len_diagrams = len(diagrams)

    diagrams[len_diagrams-4].texts[0].set_text('') #remove exciation from hydro
    diagrams[len_diagrams-5].texts[-1].set_text('') #remove excitation from excitation
    diagrams[len_diagrams-3].texts[0].set_text('') #remove absorbed from absorbed
    diagrams[len_diagrams-2].texts[0].set_text('') #remove mech from mech-elec
    diagrams[len_diagrams-2].texts[-1].set_text('') #remove electrical from mech-elec
    diagrams[len_diagrams-1].texts[0].set_text('')  #remove electrical in from elec

    if len_diagrams > 5:
        axes.legend()   #add legend for the refernce arrows
    if len_diagrams >6:
      diagrams[1].texts[0].set_text('')  

    axes.set_title(axes_title)
    axes.axis("off")

    if return_fig_and_axes:
        return fig, axes, 
