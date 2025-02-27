import autograd.numpy as np
from xarray import DataArray, Dataset, load_dataset, concat, merge
from math import comb
import wecopttool as wot
import os
import capytaine as cpy


from numpy.typing import ArrayLike
from typing import Iterable, Callable, Any, Optional, Mapping, TypeVar, Union
from autograd.numpy import ndarray
from scipy.optimize import OptimizeResult, Bounds

from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# default values
_default_parameters = {'rho': 1025.0, 'g': 9.81, 'depth': np.infty}
_default_min_damping = 1e-6

TWEC = TypeVar("TWEC", bound="WEC")
TIPP = TypeVar("TIPP", bound ="InvertedPendulumPTO")
TStateFunction = Callable[
    [TWEC, ndarray, ndarray, Dataset], ndarray]
TForceDict = dict[str, TStateFunction]
TIForceDict = Mapping[str, TStateFunction]
FloatOrArray = Union[float, ArrayLike]

class PioneerBuoy(wot.WEC):
    @staticmethod
    def from_bem_data(
        f1: int,
        nfreq: int,
        # ndof: int,
        buoy_cg: Optional[float] = 0.298, # m
        buoy_moi: Optional[float] = 7484., # kg-m^2
        draft: Optional[float] = 0.5,
        freeboard: Optional[float] = 0.516,
        data_path: Optional[str] =os.path.join('pioneer_data') ,
        f_add: Optional[TIForceDict] = None,
        constraints: Optional[Iterable[Mapping]] = None,
        min_damping: Optional[float] = _default_min_damping,
    )-> TWEC:
        bem_data_string = 'pioneer_f1_'+f'{f1}' + '_nfreq' +f'{nfreq}'+'.nc'
        bem_data_fname = os.path.join(data_path,bem_data_string)
        if os.path.isfile(bem_data_fname):
            print("Found existing BEM file, loading")
            bem_data = wot.read_netcdf(bem_data_fname)
        else:
            print("Did not find existing BEM file... \n Creating mesh and running capytaine")
            in2m = 0.0254 # inch -> meter conversion factor
            hull_geom = wot.geom.WaveBot(r1=130./2 * in2m,
                                    r2=110./2 * in2m, 
                                    h1=22.679 * in2m,
                                    h2=17.321 * in2m,
                                    scale_factor=1,
                                    freeboard=freeboard)
            mesh = hull_geom.mesh(mesh_size_factor=0.5)
            pnr_fb = cpy.FloatingBody.from_meshio(mesh, name="Pioneer")
            pnr_fb.center_of_mass = np.array([0., 0., buoy_cg])
            pnr_fb.add_rotation_dof(name='Pitch')
            pnr_fb.rotation_center = pnr_fb.center_of_mass
            ndof = pnr_fb.nb_dofs

            pnr_fb.inertia_matrix = DataArray(data=np.asarray(([[buoy_moi]])),
                                        dims=['influenced_dof', 'radiating_dof'],
                                        coords={'influenced_dof': list(pnr_fb.dofs),
                                                'radiating_dof': list(pnr_fb.dofs)},
                                        name="inertia_matrix")
            freqs = wot.frequency(f1, nfreq, False) # False -> no zero frequency                                
            bem_data = wot.run_bem(pnr_fb, freqs)
            wot.write_netcdf(bem_data_fname, bem_data)
            # pnr_fb.keep_immersed_part()
            # k_buoy = pnr_fb.compute_hydrostatic_stiffness(rho=_default_parameters['rho']).values.squeeze()
        # hd = wot.add_linear_friction(bem_data, friction = min_damping) 
        # hd = wot.check_radiation_damping(hd)
        # Zi_bem = wot.hydrodynamic_impedance(hd)
        return wot.WEC.from_bem(bem_data,
                                f_add=  f_add,
                                constraints = constraints,
                                min_damping = min_damping)

    @staticmethod
    def from_empirical_data(
        f1: int,
        nfreq: int,
        data_path: Optional[str] = 'pioneer_data' ,
        data_file: Optional[str] = 'pioneer_empirical_data.nc',
        f_add: Optional[TIForceDict] = None,
        constraints: Optional[Iterable[Mapping]] = None,
    )-> TWEC:
        full_file = os.path.join(data_path, data_file)
        empirical_data = load_dataset(full_file)
        omega_data = empirical_data.omega
        exc_coeff_data = empirical_data.exc_coeff_data_real + 1j*empirical_data.exc_coeff_data_imag
        Zi_data = empirical_data.Zi_data_real + 1j*empirical_data.Zi_data_imag
        Zi_stiffness = empirical_data.Zi_stiffness
        freqs = wot.frequency(f1, nfreq, False)
        omega = 2*np.pi*freqs
        exc_coeff_intrp = exc_coeff_data.interp(omega = omega, method='linear', kwargs={"fill_value": 0})
        Zi_intrp = Zi_data.interp(omega = omega, kwargs={"fill_value": "extrapolate"})
        return wot.WEC.from_impedance(freqs,
                                      Zi_intrp,
                                      exc_coeff_intrp,
                                      Zi_stiffness,
                                      f_add = f_add,
                                      constraints = constraints)

class InvertedPendulumPTO:
    """A base inverted pendulum power take-off (PTO) object to be used 
    in conjunction with a :py:class:`PioneerBuoy` object.
    """
    def __init__(self,
                # omega: ArrayLike,
                f1: int,
                nfreq: int,
                ndof: int,
                # nstate_pen: int,
                control_type: Optional[str] = 'unstructured',
                pendulum_moi: Optional[float] = 7.605, # kgm^2, from CAD
                pendulum_com: Optional[float] = 0.1248, #m, above pendulum shaft, via CAD
                pendulum_mass: Optional[float] = 244, # kg, via CAD
                pendulum_coulomb_friction: Optional[float] =  1.8, # N*m, coulomb friction from main bearings, in pendulum frame
                pendulum_viscous_friction: Optional[float] =  1.7, # N*ms/rad
                spring_stiffness: Optional[float] = 305,    #Nm/rad
                spring_gear_ratio: Optional[float] = 1,
                belt_gear_ratio: Optional[float] =  112/34,
                belt_power_rating: Optional[float] =  3100, #W
                gearbox_gear_ratio: Optional[float] =  6.6/1,
                gearbox_max_conti_torque:  Optional[float] =  5,     #Nm OUTPUT
                gearbox_peak_torque: Optional[float] = 18,     #Nm OUTPUT
                gearbox_moi: Optional[float] =  8.06e-6,    #kgm^2
                gearbox_friction: Optional[float] = 0.6125, #Nm #coulomb friction, in the space of the gear box output
        # fric_coul: Optional[float] = 2,
        # fric_visc: Optional[float] = 0.02,
                generator_torque_constant: Optional[float] = 0.186,   #Nm/A
                generator_winding_resistance: Optional[float] =  0.0718,   #Ohm
                generator_winding_inductance: Optional[float] = 0.0,
                generator_rotor_inertia: Optional[float] = 0.000153, #kgm^2,
                generator_max_conti_current: Optional[float] = 17.1,  #A     #max conti torque = 17.1*0.186 = 3.18Nm, on gearbox side 3.18*6.6 = 21Nm > than gear box limit
                dc_bus_max_voltage: Optional[float] = 30, #V
                generator_coulomb_friction: Optional[float] = 0.6125, #Nm #coulomb friction, in the space of the gear box output
                drivetrain_friction: Optional[float] = 0.0,
                drivetrain_stiffness: Optional[float] = 0,
                nsubsteps_constraints: Optional[int] = 4,
                name: Optional[str] = ''
        )  -> None:
        self.pto_gear_ratio = belt_gear_ratio*gearbox_gear_ratio
        freqs = wot.frequency(f1, nfreq, False)
        self.f1 = f1
        self.nfreq = nfreq
        omega = 2*np.pi*freqs
        self.omega = omega
        self.ndof = ndof
        self.control_type = control_type
        self.nfreq = nfreq
        self.nstate_pen = 2*nfreq
        self.spring_stiffness = spring_stiffness
        self.spring_gear_ratio = spring_gear_ratio
        self.pendulum_moi = pendulum_moi
        self.pendulum_com = pendulum_com
        self.pendulum_mass = pendulum_mass
        self.pendulum_coulomb_friction = pendulum_coulomb_friction
        self.pendulum_viscous_friction = pendulum_viscous_friction
        self.belt_gear_ratio = belt_gear_ratio
        self.gearbox_friction = gearbox_friction
        self.max_PTO_torque = gearbox_peak_torque*belt_gear_ratio # N*m, gearbox is more limiting than generator
        self.dc_bus_max_voltage = dc_bus_max_voltage
        self.nsubsteps_constraints = nsubsteps_constraints
        self.pto_impedance = self._pto_impedance(omega,
                        pto_gear_ratio = self.pto_gear_ratio,
                        torque_constant = generator_torque_constant,
                        drivetrain_inertia = generator_rotor_inertia+gearbox_moi,
                        drivetrain_stiffness = drivetrain_stiffness,
                        drivetrain_friction = drivetrain_friction,
                        winding_resistance = generator_winding_resistance,
                        winding_inductance = generator_winding_inductance)
        self.pto_transfer_mat = self._pto_transfer_mat(self.pto_impedance)
        # self.torque_from_PTO, self.nstate_pto, self.nstate_opt =  self._create_control_torque_function
        self.nstate_pto, self.nstate_opt = self.nstate_based_on_control()
        self.name = name

    def _pto_impedance(self,
        omega,
        pto_gear_ratio,
        torque_constant,
        drivetrain_inertia,
        drivetrain_stiffness,
        drivetrain_friction,
        winding_resistance,
        winding_inductance
    ):
        drivetrain_impedance = (1j*omega*drivetrain_inertia +
                            drivetrain_friction +
                            1/(1j*omega)*drivetrain_stiffness)

        winding_impedance = winding_resistance + 1j*omega*winding_inductance


        pto_impedance_11 = -1* pto_gear_ratio**2 * drivetrain_impedance
        off_diag = np.sqrt(3.0/2.0) * torque_constant * pto_gear_ratio
        pto_impedance_12 = -1*(off_diag+0j) * np.ones(omega.shape)
        pto_impedance_21 = -1*(off_diag+0j) * np.ones(omega.shape)
        pto_impedance_22 = winding_impedance
        pto_impedance = np.array([[pto_impedance_11, pto_impedance_12],
                                [pto_impedance_21, pto_impedance_22]])
        return pto_impedance

    def _pto_transfer_mat(self, 
                          pto_impedance
    ):
        pto_impedance_abcd = wot.pto._make_abcd(pto_impedance, ndof=self.ndof)
        pto_transfer_mat = wot.pto._make_mimo_transfer_mat(pto_impedance_abcd,
                                         ndof=self.ndof)
        return pto_transfer_mat   

    def x_pen(self, wec, x_wec, x_opt, waves, nsubsteps=1):                               
        x_pos_pen = wec.vec_to_dofmat(x_opt[self.nstate_pto:])
        return x_pos_pen

    def x_rel(self, wec, x_wec, x_opt, waves, nsubsteps=1):
        x_pos_buoy = wec.vec_to_dofmat(x_wec)
        x_pos_pen = self.x_pen(wec, x_wec, x_opt, waves, nsubsteps)
        return x_pos_buoy - x_pos_pen

    def pen_position(self, wec, x_wec, x_opt, waves, nsubsteps=1):
        x_pos_pen = self.x_pen(wec, x_wec, x_opt, waves, nsubsteps)
        time_matrix = wec.time_mat_nsubsteps(nsubsteps)
        return np.dot(time_matrix, x_pos_pen)

    def pen_velocity(self, wec, x_wec, x_opt, waves, nsubsteps=1):
        x_pos_pen = self.x_pen(wec, x_wec, x_opt, waves, nsubsteps)
        x_vel_pen = np.dot(wec.derivative_mat, x_pos_pen)
        time_matrix = wec.time_mat_nsubsteps(nsubsteps)
        return np.dot(time_matrix, x_vel_pen)

    def rel_position(self, wec, x_wec, x_opt, waves, nsubsteps=1):
        pos_rel = self.x_rel(wec, x_wec, x_opt, waves)
        time_matrix = wec.time_mat_nsubsteps(nsubsteps)
        return np.dot(time_matrix, pos_rel)

    def rel_velocity(self, wec, x_wec, x_opt, waves, nsubsteps=1):
        x_pos_rel = self.x_rel(wec, x_wec, x_opt, waves)
        x_vel_rel = np.dot(wec.derivative_mat, x_pos_rel)
        time_matrix = wec.time_mat_nsubsteps(nsubsteps)
        return np.dot(time_matrix, x_vel_rel)
    ## PTO torque depending on controller

    

    def torque_from_PTO(self, wec, x_wec, x_opt, waves=None, nsubsteps=1):
        # Call the appropriate torque calculation method based on the control type
        if self.control_type == 'unstructured':
            return self.torque_from_unstructured(wec, x_wec, x_opt, waves, nsubsteps)
        elif self.control_type == 'P':
            return self.torque_from_damping(wec, x_wec, x_opt, waves, nsubsteps)
        elif self.control_type == 'PI':
            return self.torque_from_PI(wec, x_wec, x_opt, waves, nsubsteps)
        elif self.control_type == 'I':
            return self.torque_from_I(wec, x_wec, x_opt, waves, nsubsteps)            
        elif self.control_type == 'PID':
            return self.torque_from_PID(wec, x_wec, x_opt, waves, nsubsteps)
        elif self.control_type == 'nonlinear_3rdO':
            return self.torque_from_nonlinear_3rdO(wec, x_wec, x_opt, waves, nsubsteps)
        else:
            raise ValueError("Invalid control type. Choose 'unstructured', 'P', 'PI', 'I', 'PID', or 'nonlinear_3rdO'.")

    def nstate_based_on_control(self):
        control_type = self.control_type
        nfreq = self.nfreq
        nstate_pen = self.nstate_pen
        if control_type == 'unstructured':
            nstate_pto = 2 * nfreq
            nstate_opt = nstate_pto + nstate_pen
            return nstate_pto, nstate_opt
        elif control_type == 'P':
            nstate_pto = 1
            nstate_opt = nstate_pto + nstate_pen
            return nstate_pto, nstate_opt
        elif control_type == 'PI':
            nstate_pto = 2
            nstate_opt = nstate_pto + nstate_pen
            return nstate_pto, nstate_opt
        elif control_type == 'I':
            nstate_pto = 1
            nstate_opt = nstate_pto + nstate_pen
            return nstate_pto, nstate_opt            
        elif control_type == 'PID':
            nstate_pto = 3
            nstate_opt = nstate_pto + nstate_pen
            return nstate_pto, nstate_opt
        elif control_type == 'nonlinear_3rdO':
            nstate_pto = 7
            nstate_opt = nstate_pto + nstate_pen
            return nstate_pto, nstate_opt
        else:
            raise ValueError("Invalid control type. Choose 'unstructured', 'P', 'PI', 'PID', or 'nonlinear_3rdO'.")

    def bounds_based_on_control(self):
        control_type = self.control_type
        unbound_list = self.nstate_opt*[np.Infinity]
        lb_list = (-1*np.array(unbound_list)).tolist()
        if control_type == 'unstructured':
            return None
        elif control_type == 'P':
            return None
        elif control_type == 'PI':
            return None
        elif control_type == 'I':
            return None
        elif control_type == 'PID':
            lb_list[2] = 0
            bounds_PID = Bounds(lb=lb_list, ub=unbound_list)
            return bounds_PID
        elif control_type == 'nonlinear_3rdO':
            return None
        else:
            raise ValueError("Invalid control type. Choose 'unstructured', 'P', 'PI', 'PID', or 'nonlinear_3rdO'.")

    def torque_from_unstructured(self, wec, x_wec, x_opt, waves=None, nsubsteps=1):
        f_fd = np.reshape(x_opt[:self.nstate_pto], (-1, self.ndof), order='F')  # Take the first components for PTO torque
        time_matrix = wec.time_mat_nsubsteps(nsubsteps)
        torque = np.dot(time_matrix, f_fd)  
        return torque

    def torque_from_damping(self, wec, x_wec, x_opt, waves=None, nsubsteps=1):
        pos_rel = self.x_rel(wec, x_wec, x_opt, waves)
        vel_rel = np.dot(wec.derivative_mat, pos_rel)
        f_fd = x_opt[:self.nstate_pto] * vel_rel
        time_matrix = wec.time_mat_nsubsteps(nsubsteps)
        torque = np.dot(time_matrix, f_fd)  
        return torque

    def torque_from_PI(self, wec, x_wec, x_opt, waves=None, nsubsteps=1):
        pos_rel = self.x_rel(wec, x_wec, x_opt, waves)
        vel_rel = np.dot(wec.derivative_mat, pos_rel)
        f_fd = self.torque_PI(vel_rel, pos_rel, x_opt[:self.nstate_pto])
        time_matrix = wec.time_mat_nsubsteps(nsubsteps)
        torque = np.dot(time_matrix, f_fd)
        return torque

    def torque_from_I(self, wec, x_wec, x_opt, waves=None, nsubsteps=1):
        pos_rel = self.x_rel(wec, x_wec, x_opt, waves)
        # vel_rel = np.dot(wec.derivative_mat, pos_rel)
        f_fd = self.torque_I(pos_rel, x_opt[:self.nstate_pto])
        time_matrix = wec.time_mat_nsubsteps(nsubsteps)
        torque = np.dot(time_matrix, f_fd)
        return torque        

    def torque_from_PID(self, wec, x_wec, x_opt, waves=None, nsubsteps=1):
        pos_rel = self.x_rel(wec, x_wec, x_opt, waves)
        vel_rel = np.dot(wec.derivative_mat, pos_rel)
        acc_rel = np.dot(wec.derivative_mat, vel_rel)
        f_fd = self.torque_PID(vel_rel, pos_rel, acc_rel, x_opt[:self.nstate_pto])
        time_matrix = wec.time_mat_nsubsteps(nsubsteps)
        torque = np.dot(time_matrix, f_fd)
        return torque

    def torque_from_nonlinear_3rdO(self, wec, x_wec, x_opt, waves=None, nsubsteps=1):
        pos_rel = self.x_rel(wec, x_wec, x_opt, waves)
        vel_rel = np.dot(wec.derivative_mat, pos_rel)
        f_fd = self.torque_3rd_polynomial(vel_rel, pos_rel, x_opt[:self.nstate_pto])
        time_matrix = wec.time_mat_nsubsteps(nsubsteps)
        torque = np.dot(time_matrix, f_fd) 
        return torque

    def torque_I(self, pos, coeffs):
        return (coeffs[0] * pos)

    def torque_PI(self, vel, pos, coeffs):
        return (coeffs[0] * vel +  
                coeffs[1] * pos)

    def torque_PID(self, vel, pos, acc, coeffs):
        return (coeffs[0] * vel +  
                coeffs[1] * pos +
                coeffs[2] * acc)

    def torque_3rd_polynomial(self, vel, pos, coeffs):
        return (coeffs[0] +   
                coeffs[1] * vel +   
                coeffs[2] * pos +   
                coeffs[3] * vel**3 +   
                coeffs[4] * pos**3 +  
                coeffs[5] * vel**2 * pos +  
                coeffs[6] * vel * pos**2)  

    ## additional torque
    def torque_from_friction(self, wec, x_wec, x_opt, waves=None, nsubsteps=1):
        raise NotImplementedError("This method should be implemented in subclasses.")
    def torque_from_spring(self, wec, x_wec, x_opt, waves=None, nsubsteps=1):
        raise NotImplementedError("This method should be implemented in subclasses.")
    def torque_from_pendulum(self, wec, x_wec, x_opt, waves=None, nsubsteps=1):
        raise NotImplementedError("This method should be implemented in subclasses.")


    def pendulum_inertia(self, wec, x_wec, x_opt, waves = None, nsubsteps = 1):
        # pos_pen = wec.vec_to_dofmat(x_opt[self.nstate_pto:])
        x_pos_pen = self.x_pen(wec, x_wec, x_opt, waves, nsubsteps)
        acc_pen = np.dot(wec.derivative2_mat, x_pos_pen)
        time_matrix = wec.time_mat_nsubsteps(nsubsteps)
        acc_pen = np.dot(time_matrix, acc_pen)
        return self.pendulum_moi * acc_pen
    
    ## objective function
    
    def mechanical_power(self, wec, x_wec, x_opt, waves, nsubsteps=1):
        torque_td = self.torque_from_PTO(wec, x_wec, x_opt, waves, nsubsteps)
        vel_td = self.rel_velocity(wec, x_wec, x_opt, waves, nsubsteps)
        return vel_td * torque_td

    def power_variables(self, wec, x_wec, x_opt, waves, nsubsteps=1):
        q1_td = self.rel_velocity(wec, x_wec, x_opt, waves)
        e1_td = self.torque_from_PTO(wec, x_wec, x_opt, waves)
        q1 = wot.complex_to_real(wec.td_to_fd(q1_td, False))
        e1 = wot.complex_to_real(wec.td_to_fd(e1_td, False))
        vars_1 = np.hstack([q1, e1])
        vars_1_flat = wec.dofmat_to_vec(vars_1)
        vars_2_flat = np.dot(self.pto_transfer_mat, vars_1_flat)
        vars_2 = wot.vec_to_dofmat(vars_2_flat, 2)
        q2 = vars_2[:, 0]
        e2 = vars_2[:, 1]
        time_mat = wec.time_mat_nsubsteps(nsubsteps)
        q2_td = np.dot(time_mat, q2)
        e2_td = np.dot(time_mat, e2)
        return q2_td, e2_td

    def electrical_power(self, wec, x_wec, x_opt, waves, nsubsteps=1):
        # q1_td = self.rel_velocity(wec, x_wec, x_opt, waves)
        # e1_td = self.torque_from_PTO(wec, x_wec, x_opt, waves)
        # q1 = wot.complex_to_real(wec.td_to_fd(q1_td, False))
        # e1 = wot.complex_to_real(wec.td_to_fd(e1_td, False))
        # vars_1 = np.hstack([q1, e1])
        # vars_1_flat = wec.dofmat_to_vec(vars_1)
        # vars_2_flat = np.dot(self.pto_transfer_mat, vars_1_flat)
        # vars_2 = wot.vec_to_dofmat(vars_2_flat, 2)
        # q2 = vars_2[:, 0]
        # e2 = vars_2[:, 1]
        # time_mat = wec.time_mat_nsubsteps(nsubsteps)
        # q2_td = np.dot(time_mat, q2)
        # e2_td = np.dot(time_mat, e2)
        q2_td, e2_td = self.power_variables(wec, x_wec,
                                            x_opt, waves, nsubsteps)
        epower_td = q2_td * e2_td                             
        return np.expand_dims(epower_td, axis=1)

    def back_emf(self, wec, x_wec, x_opt, waves, nsubsteps=1):
        q2_td, e2_td = self.power_variables(wec, x_wec,
                                            x_opt, waves, nsubsteps)
        return e2_td

    def quad_current(self, wec, x_wec, x_opt, waves, nsubsteps=1):
        q2_td, e2_td = self.power_variables(wec, x_wec,
                                            x_opt, waves, nsubsteps)
        return q2_td

    def energy(self, wec, x_wec, x_opt, waves, nsubsteps=1):
        power_td = self.electrical_power(wec, x_wec, x_opt, waves, nsubsteps)
        return np.sum(power_td) * wec.dt/nsubsteps

    def average_electrical_power(self, wec, x_wec, x_opt, waves, nsubsteps=1):
        e = self.energy(wec, x_wec, x_opt, waves, nsubsteps)
        return e / wec.tf



    def solve(self, 
              wec,
              waves: Dataset,
              x_wec_0: Optional[ndarray] = None,
              x_opt_0: Optional[ndarray] = None,
              bounds_opt: Optional[Bounds] = None,
              max_attempts: Optional[float]  = 3,
              **kwargs)-> list[OptimizeResult]:
        if bounds_opt is None:
            bounds_opt = self.bounds_based_on_control()
        attempts = 0
        while attempts < max_attempts:
            try:
                res = wec.solve(waves,
                        obj_fun = self.average_electrical_power,
                        nstate_opt = self.nstate_opt,
                        optim_options={'maxiter': 200,
                                       'disp':False},
                        x_wec_0=x_wec_0, # initialize with result from linearized case
                        x_opt_0=x_opt_0, # initialize with result from linearized case
                        scale_x_wec=1e1,
                        scale_x_opt=np.concatenate((np.array([1e-1])*np.ones(self.nstate_pto), 1e1 * np.ones(self.nstate_pen))),
                        scale_obj=1e-1,
                        bounds_opt= bounds_opt,
                        **kwargs)
                status_list = []
                for idx, ires in enumerate(res):
                    print(f"{self.name} wave {idx}, exit mode: {ires.status}, nit: {ires.nit}, cntr: {self.control_type}, avg. power: {ires.fun:.2f}W")
                    status_list.append(ires.status)
                if all(x == 0 for x in status_list):
                    # print("Operation successful, exiting.")
                    return  res# Exit the function if successful
                elif any(x == 9 for x in status_list):
                    print("Exit mode 9 encountered, trying again...")
                else:
                    print(f"Unexpected exit mode, trying again...")
            except Exception as e:
                print(f"An error occurred: {e}, trying again...")
            attempts += 1
        
        return res

    def _postproc(self,
                    wec,
                    res_opt,
                    waves,
                    nsubsteps) -> tuple[list[Dataset], list[Dataset]]:
        """Post process of single results (not list)"""
        x_wec, x_opt = wec.decompose_state(res_opt.x)
        t_dat = wec.time_nsubsteps(nsubsteps)

        rel_vel_td = self.rel_velocity(wec, x_wec, x_opt, waves, nsubsteps)
        rel_vel_fd = wec.td_to_fd(rel_vel_td[::nsubsteps])
        rel_vel_attr = {'long_name': 'Relative velocity', 'units': 'rad/s'}

        rel_pos_td = self.rel_position(wec, x_wec, x_opt, waves, nsubsteps)
        rel_pos_fd = wec.td_to_fd(rel_pos_td[::nsubsteps])
        rel_pos_attr = {'long_name': 'Relative position', 'units': 'rad'}

        pen_pos_td = self.pen_position(wec, x_wec, x_opt, waves, nsubsteps)
        pen_pos_fd = wec.td_to_fd(pen_pos_td[::nsubsteps])
        pen_pos_attr = {'long_name': 'Pendulum position', 'units': 'rad'}

        pen_vel_td = self.pen_velocity(wec, x_wec, x_opt, waves, nsubsteps)
        pen_vel_fd = wec.td_to_fd(pen_vel_td[::nsubsteps])
        pen_vel_attr = {'long_name': 'Pendulum velocity', 'units': 'rad/s'}

        mpower_td = self.mechanical_power(wec, x_wec, x_opt, waves, nsubsteps)
        mpower_fd = wec.td_to_fd(mpower_td[::nsubsteps])
        mpower_attr = {'long_name': 'Mechanical power', 'units': 'W'}

        epower_td = self.electrical_power(wec, x_wec, x_opt, waves, nsubsteps)
        epower_fd = wec.td_to_fd(epower_td[::nsubsteps])
        epower_attr = {'long_name': 'Electrical power', 'units': 'W'}

        back_emf_td = np.expand_dims(self.back_emf(wec, x_wec, x_opt, waves, nsubsteps),axis = 1)
        back_emf_fd = wec.td_to_fd(back_emf_td[::nsubsteps])
        back_emf_attr = {'long_name': 'Back electromotive force', 'units': 'V'}

        quad_cur_td = np.expand_dims(self.quad_current(wec, x_wec, x_opt, waves, nsubsteps),axis = 1)
        quad_cur_fd = wec.td_to_fd(quad_cur_td[::nsubsteps])
        quad_cur_attr = {'long_name': 'Quadrature current', 'units': 'A'}

        names = ["relative PTO Dof"]

        omega_attr = {'long_name': 'Radial frequency', 'units': 'rad/s'}
        freq_attr = {'long_name': 'Frequency', 'units': 'Hz'}
        period_attr = {'long_name': 'Period', 'units': 's'}
        dof_attr = {'long_name': 'PTO degree of freedom'}
        time_attr = {'long_name': 'Time', 'units': 's'}
        torque_attr = {'long_name': 'Torque', 'units': 'Nm'}

        coords_fd = {'omega':('omega', wec.omega, omega_attr),
                'freq':('omega', wec.frequency, freq_attr),
                'period':('omega', wec.period, period_attr),
                'dof':('dof', names, dof_attr),         }

        coords_td = {'time':('time', t_dat, time_attr),
                'dof':('dof', names, dof_attr),        }


        pen_fd_state = Dataset(
            data_vars={
                'pen_pos': (['omega','dof'], pen_pos_fd, pen_pos_attr),
                'rel_vel': (['omega','dof'], rel_vel_fd, rel_vel_attr),
                'rel_pos': (['omega','dof'], rel_pos_fd, rel_pos_attr),
                'pen_vel': (['omega','dof'], pen_vel_fd, pen_vel_attr),
                'epower': (['omega','dof'], epower_fd, epower_attr),
                'mpower': (['omega','dof'], mpower_fd, mpower_attr),
                'back_emf': (['omega','dof'], back_emf_fd, back_emf_attr),
                'quad_current': (['omega','dof'], quad_cur_fd, quad_cur_attr),

            },
            coords=coords_fd,
            attrs={},
        )

        pen_td_state = Dataset(
            data_vars={
                'pen_pos': (['time','dof'], pen_pos_td, pen_pos_attr),
                'rel_vel': (['time','dof'], rel_vel_td, rel_vel_attr),
                'rel_pos': (['time','dof'], rel_pos_td, rel_pos_attr),
                'pen_vel': (['time','dof'], pen_vel_td, pen_vel_attr),
                'epower': (['time', 'dof'], epower_td, epower_attr),
                'mpower': (['time', 'dof'], mpower_td, mpower_attr),
                'back_emf': (['time','dof'], back_emf_td, back_emf_attr),
                'quad_current': (['time','dof'], quad_cur_td, quad_cur_attr),

            },
            coords=coords_td,
            attrs={}
        )

        torque_td_da_list = []
        torque_fd_da_list = []

        for name, torque in self.f_add.items():
            torque_td = torque(wec, x_wec, x_opt, waves, nsubsteps)
            torque_fd = wec.td_to_fd(torque_td[::nsubsteps])  # no substeps
            torque_td_da = DataArray(data = torque_td,
                        dims = ['time','dof'],
                        coords = coords_td,
                        attrs = torque_attr
                            ).expand_dims({'type': [name]})
            torque_fd_da = DataArray(data = torque_fd,
                        dims = ['omega','dof'],
                        coords = coords_fd,
                        attrs = torque_attr
                            ).expand_dims({'type': [name]})
            torque_td_da_list.append(torque_td_da)
            torque_fd_da_list.append(torque_fd_da)

        pen_torque_td = concat(torque_td_da_list, dim = 'type')
        pen_torque_td.type.attrs['long_name'] = 'Type'
        pen_torque_td.name = 'torque'
        pen_torque_fd = concat(torque_fd_da_list, dim = 'type')
        pen_torque_fd.type.attrs['long_name'] = 'Type'
        pen_torque_fd.name = 'torque'
        
        
        # pen_fdom = merge(pen_fd_state)
        pen_fdom = merge([pen_fd_state,pen_torque_fd])
        pen_tdom = merge([pen_td_state,pen_torque_td])

        return pen_fdom, pen_tdom

    def post_process(self,
                        wec,
                        res_opt,
                        waves,
                        nsubsteps):
        wec_fdom, wec_tdom = wec.post_process(wec, res_opt, waves, nsubsteps=nsubsteps)

        pen_fdom = []
        pen_tdom = []
        for idx, ires in enumerate(res_opt):
            ifd, itd = self._postproc(wec, ires, waves.sel(realization=idx), nsubsteps)
            pen_fdom.append(ifd)
            pen_tdom.append(itd)
        return wec_fdom, wec_tdom, pen_fdom, pen_tdom 

    # def animate_results(self,
    #                     wec_tdom: Dataset,
    #                     pen_tdom: Dataset)

    def animate_results(self,
                        wec_tdom: Dataset,
                        pen_tdom: Dataset,
                        waves:Dataset):
        plt.rcParams["animation.html"] = "jshtml"
        plt.ioff()
        fig, ax = plt.subplots()

        # Set the limits of the plot
        xlim = [-1.2, 1.2]
        dx = xlim[1]-xlim[0]
        ylim0 = -0.8
        ylim = [ylim0, ylim0+dx]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        wave_number_deep = waves.omega**2/9.81

        spatial_x = np.linspace(xlim[0], xlim[1], 10)
        wave_elevations = np.zeros((len(spatial_x), len(pen_tdom['time'])))

        #phase shift waves
        for i, x in enumerate(spatial_x):
            manual_wave = wot.waves.elevation_fd(self.f1, self.nfreq, 
                                                directions=waves.wave_direction, 
                                                nrealizations=len(waves.realization), 
                                                amplitudes=np.abs(waves), 
                                                phases=np.rad2deg(np.angle(waves) - np.expand_dims(wave_number_deep*x,axis=(1, 2))))    
            wave_td = wot.time_results(manual_wave, pen_tdom['time'])
            wave_elevations[i, :] = wave_td[0, 0, :]

        #Inputs NPIP, or some Pen pen_tdom, wec_tdom

        frames = len(pen_tdom['time'])
        time = pen_tdom['time']
        wave_elev = wec_tdom['wave_elev'].squeeze()
        pendulum_angles = pen_tdom['pen_pos'].squeeze()  # Sine wave for the circle
        buoy_angles = wec_tdom['pos'].squeeze()
        pendulum_max_torque = (self.pendulum_mass * _default_parameters['g'] * self.pendulum_com )
        pendulum_torque_norm = pen_tdom['torque'].sel(type = 'Pendulum NL').squeeze() / pendulum_max_torque
        spring_torque_norm = pen_tdom['torque'].sel(type = 'Spring NL').squeeze() / pendulum_max_torque# 614.7460754866761  #max value numericall from the acutal used function
        friction_torque_rel = pen_tdom['torque'].sel(type = 'Friction NL').squeeze() /pendulum_max_torque


        pto_torque = pen_tdom['torque'].sel(type = 'Generator').squeeze()
        pto_torque_rel = pto_torque / pendulum_max_torque
        pow_norm_factor = 200
        pto_elec_power = pen_tdom['epower'].squeeze()
        pto_elec_power_norm = pto_elec_power / pow_norm_factor

        # Rectangle and truncated cone parameters
        buoy_width = 1.8
        buoy_height = 0.6
        cone_height = 0.6
        cone_radius_top = buoy_width / 4  # Top radius matches half the rectangle width
        cone_radius_bottom = buoy_width /2 # Bottom radius matches the rectangle width

            
        delta_y = 0.2

        #zero pos vectors
        buoy_x = np.array([-buoy_width / 2, buoy_width / 2, buoy_width / 2, -buoy_width / 2, -buoy_width / 2])
        buoy_y = np.array([-buoy_height / 2, -buoy_height / 2, buoy_height / 2, buoy_height / 2, -buoy_height / 2]) + delta_y
        buoy_initial =  np.vstack((buoy_x, buoy_y))  

        cone_x = np.linspace(-cone_radius_bottom, cone_radius_bottom, 2)
        cone_y_top = np.full_like(cone_x, -buoy_height / 2) +delta_y # Top of the cone
        cone_y_bottom = cone_y_top - cone_height +delta_y # Bottom of the cone
        cone_top = np.vstack((cone_x, cone_y_top))
        cone_bottom = np.vstack((cone_x * (cone_radius_top / cone_radius_bottom), cone_y_bottom))
        
        cone_left_side = np.array([[cone_x[0], cone_y_top[0]], [cone_x[0] * (cone_radius_top / cone_radius_bottom), cone_y_bottom[0]]])
        cone_right_side = np.array([[cone_x[-1], cone_y_top[-1]], [cone_x[-1] * (cone_radius_top / cone_radius_bottom), cone_y_bottom[-1]]])
        
        len_pen = 0.5
        def torque_to_arc(pen_torque_norm):
            return (0, 90 * pen_torque_norm) if pen_torque_norm >= 0 else (90 * pen_torque_norm, 0)
        def plot_torque(torque, label, radius, color):
            th1, th2 = torque_to_arc(torque)
            arc = patches.Arc((0, 0), radius, radius, angle=90, theta1=th1, theta2=th2, color=color, linewidth=2, alpha = 0.5, label =label)
            ax.add_patch(arc)

        def animate(frame):
            plt.cla()  # Clear the current axes
            ax.plot(spatial_x, wave_elevations[:, frame], color='b')
            # Get the predefined angles for the current frame
            angle_pen = pendulum_angles[frame]
            angle_buoy = buoy_angles[frame]
            torque_pen = pendulum_torque_norm[frame].item()
            torque_spring = spring_torque_norm[frame].item()
            torque_friction = friction_torque_rel[frame].item()
            # torque_pen_inertia = pendulum_inerta_torque_norm[frame].item()

            torque_pto = pto_torque_rel[frame].item()
            power_pto = pto_elec_power_norm[frame].item()

            # Calculate the x and y coordinates of the pendulum
            x_pen = len_pen * np.sin(angle_pen)
            y_pen = len_pen * np.cos(angle_pen)

            
            # pendulum
            plt.plot([0, x_pen], [0, y_pen], 'r--')  # Dashed red line
            plt.plot(x_pen, y_pen, 'ro', markersize = 12)  # 'ro' means red color, circle marker

            # Rotate the rectangle using the predefined angle
            rotation_matrix = np.array([[np.cos(angle_buoy), -np.sin(angle_buoy)],
                                        [np.sin(angle_buoy), np.cos(angle_buoy)]])
            buoy_rotated = rotation_matrix @ buoy_initial
            plt.plot(buoy_rotated[0, :], buoy_rotated[1, :], color='green')  # Green rectangle
            

            # Rotate the truncated cone
            cone_rotated_top = rotation_matrix @ cone_top
            cone_rotated_bottom = rotation_matrix @ cone_bottom
            left_side_rotated = rotation_matrix @ cone_left_side.T
            right_side_rotated = rotation_matrix @ cone_right_side.T
            
            # Plot the truncated cone
            plt.plot(cone_rotated_top[0, :], cone_rotated_top[1, :], color='green')  # Top of the cone
            plt.plot(cone_rotated_bottom[0, :], cone_rotated_bottom[1, :], color='green')  # Bottom of the cone
            plt.plot(left_side_rotated[0, :], left_side_rotated[1, :], color='green')  # Left side of the cone
            plt.plot(right_side_rotated[0, :], right_side_rotated[1, :], color='green')  # Right side of the cone

            #power bar
            if power_pto < 0:
                plt.plot([1.1, 1.1],[0, 0-power_pto],  color='green', linewidth=10, alpha = 0.5)
            else:
                plt.plot([1.1, 1.1],[0, 0-power_pto],  color='red', linewidth=10, alpha = 0.5)
            plt.text(1.2, 0, f'{-1*power_pto*pow_norm_factor:.1f}W')
            
            #torques
            plot_torque(torque_pen, label='pen. gravit.',radius= 0.5, color='red')
            plot_torque(torque_spring, label='Spring',radius= 0.6, color='blue')
            plot_torque(torque_pto, label='PTO',radius= 0.4, color='orange')
            plot_torque(torque_friction, label='Friction',radius= 0.3, color='black')
    

            # Set the limits and aspect ratio
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect('equal', adjustable='box')  
            ax.grid(True)  
            ax.legend(loc = 'upper left')
            ax.set_title(f'Time = {time[frame]:.2f}')  
        return FuncAnimation(fig, animate, frames=frames, interval=int(1000*(time[1]-time[0])))

class NonlinearInvertedPendulumPTO(InvertedPendulumPTO):
    """A nonlinear inverted pendulum power take-off (PTO) object to be used 
    in conjunction with a :py:class:`PioneerBuoy` object.
    """
    def __init__(self, f1: int, nfreq: int, ndof: int, **kwargs):
        super().__init__(f1, nfreq, ndof, name = 'NonLin', **kwargs)
        self.f_add = {
            'Generator': self.torque_from_PTO,
            'Friction NL': self.torque_from_friction,
            'Spring NL': self.torque_from_spring,
            'Pendulum NL': self.torque_from_pendulum,
        }

        self.constraints = [
            {'type': 'eq', 'fun': self.pendulum_residual}, # pendulum EoM
            {'type': 'ineq', 'fun': self.constraint_max_generator_torque},
            {'type': 'ineq', 'fun': self.constraint_max_dc_bus_voltage},

        ]

    def torque_from_friction(self, wec, x_wec, x_opt, waves, nsubsteps = 1):
        #nonlinear
        rel_vel = self.rel_velocity(wec, x_wec, x_opt, waves, nsubsteps)
        #generator and gearbox have Clolumb friction, the we'll convert into the relative vel frame
        combined_Coulomg_friction = (self.pendulum_coulomb_friction + 
                            self.gearbox_friction*self.belt_gear_ratio)
        fric =  -1*(np.tanh(rel_vel)*combined_Coulomg_friction + rel_vel*self.pendulum_viscous_friction) 
        return fric

    def nonlinear_spring_torque(self, spring_pos):
        # 135 deg nonlinear spring
        spring_eq_pos_td = spring_pos - np.pi
        n = 12
        slope = 1/(2**(2*n))*comb(2*n,n)
        scale = 1/slope
        new_pos = 0
        for ind in range(n):
            k = ind+1
            coeffs = comb(2*n, n-k)/(k*(2**(2*n-1)))
            new_pos = new_pos - coeffs*np.sin(k*spring_eq_pos_td)
        return  -1*self.spring_stiffness * scale * new_pos

    def torque_from_spring(self, wec, x_wec, x_opt, waves, nsubsteps = 1):
        #nonlinear
        rel_pos = self.rel_position(wec, x_wec, x_opt, waves, nsubsteps) 
        spring_pos = self.spring_gear_ratio * rel_pos
        spring_torque = self.nonlinear_spring_torque(spring_pos)
        spring_torque_on_shaft = self.spring_gear_ratio * spring_torque
        return spring_torque_on_shaft

    def torque_from_pendulum(self, wec, x_wec, x_opt, waves, nsubsteps=1):
        #nonlinear
        # pos_pen = wec.vec_to_dofmat(x_opt[self.nstate_pto:])
        x_pos_pen = self.x_pen(wec, x_wec, x_opt, waves, nsubsteps)
        time_matrix = wec.time_mat_nsubsteps(nsubsteps)
        pos_pen = np.dot(time_matrix, x_pos_pen)
        return -1*self.pendulum_mass * _default_parameters['g'] * self.pendulum_com * np.sin(pos_pen)
    ## constraints
        #TODO: How to pass substeps?
    def constraint_max_generator_torque(self, wec, x_wec, x_opt, waves, nsubsteps = 5):
        torque = self.torque_from_PTO(wec, x_wec, x_opt, waves, nsubsteps)
        return self.max_PTO_torque - np.abs(torque.flatten())
    def constraint_max_dc_bus_voltage(self, wec, x_wec, x_opt, waves, nsubsteps = 5):
        back_emf = self.back_emf(wec, x_wec, x_opt, waves, nsubsteps)
        return self.dc_bus_max_voltage - np.abs(back_emf.flatten())

    ## residual
    def pendulum_residual(self, wec, x_wec, x_opt, waves = None, nsubsteps = 1):
        resid = (
        self.pendulum_inertia(wec, x_wec, x_opt, waves, nsubsteps) +
        self.torque_from_pendulum(wec, x_wec, x_opt, waves, nsubsteps) +
        self.torque_from_spring(wec, x_wec, x_opt, waves, nsubsteps) +
        self.torque_from_friction(wec, x_wec, x_opt, waves, nsubsteps) +
        self.torque_from_PTO(wec, x_wec, x_opt, waves, nsubsteps)
        )
        return resid.flatten()

class LinearizedInvertedPendulumPTO(InvertedPendulumPTO):
    """A linearzied inverted pendulum power take-off (PTO) object to be used 
    in conjunction with a :py:class:`PioneerBuoy` object.
    """
    def __init__(self, f1: int, nfreq: int, ndof: int, **kwargs):
        super().__init__(f1, nfreq, ndof, name = 'Linearized', **kwargs)
        self.f_add = {
            'Generator': self.torque_from_PTO,
            'Friction': self.torque_from_friction,
            'Spring': self.torque_from_spring,
            'Pendulum': self.torque_from_pendulum,
        }

        self.constraints = [
            {'type': 'eq', 'fun': self.pendulum_residual}, # pendulum EoM
            {'type': 'ineq', 'fun': self.constraint_max_generator_torque},
            {'type': 'ineq', 'fun': self.constraint_max_dc_bus_voltage},        
        ]

    def torque_from_friction(self, wec, x_wec, x_opt, waves, nsubsteps = 1):
        #linear
        rel_vel = self.rel_velocity(wec, x_wec, x_opt, waves, nsubsteps)
        fric =  -2* rel_vel*self.pendulum_viscous_friction #increased viscous fric becuase no Coulomb
        return fric

    def torque_from_spring(self, wec, x_wec, x_opt, waves, nsubsteps = 1):
        #linear
        rel_pos = self.rel_position(wec, x_wec, x_opt, waves, nsubsteps) 
        spring_pos = self.spring_gear_ratio * rel_pos
        linear_spring_torque = -1*self.spring_stiffness*spring_pos
        linear_spring_torque_on_shaft = self.spring_gear_ratio * linear_spring_torque
        return linear_spring_torque_on_shaft

    def torque_from_pendulum(self, wec, x_wec, x_opt, waves, nsubsteps=1):
        #linear
        # pos_pen = wec.vec_to_dofmat(x_opt[self.nstate_pto:])
        x_pos_pen = self.x_pen(wec, x_wec, x_opt, waves, nsubsteps)
        time_matrix = wec.time_mat_nsubsteps(nsubsteps)
        pos_pen = np.dot(time_matrix, x_pos_pen)
        return -1*self.pendulum_mass * _default_parameters['g'] * self.pendulum_com * pos_pen
    ## constraints
        #TODO: How to pass substeps?
    def constraint_max_generator_torque(self, wec, x_wec, x_opt, waves, nsubsteps = 5):
        torque = self.torque_from_PTO(wec, x_wec, x_opt, waves, nsubsteps)
        return self.max_PTO_torque - np.abs(torque.flatten())
    def constraint_max_dc_bus_voltage(self, wec, x_wec, x_opt, waves, nsubsteps = 5):
        back_emf = self.back_emf(wec, x_wec, x_opt, waves, nsubsteps)
        return self.dc_bus_max_voltage - np.abs(back_emf.flatten())
    ## residual
    def pendulum_residual(self, wec, x_wec, x_opt, waves = None, nsubsteps = 1):
        resid = (
        self.pendulum_inertia(wec, x_wec, x_opt, waves, nsubsteps) +
        self.torque_from_pendulum(wec, x_wec, x_opt, waves, nsubsteps) +
        self.torque_from_spring(wec, x_wec, x_opt, waves, nsubsteps) +
        self.torque_from_friction(wec, x_wec, x_opt, waves, nsubsteps) +
        self.torque_from_PTO(wec, x_wec, x_opt, waves, nsubsteps)
        )
        return resid.flatten()