import autograd.numpy as np
from xarray import DataArray, Dataset, load_dataset
from math import comb
import wecopttool as wot
import os
import capytaine as cpy


from numpy.typing import ArrayLike
from typing import Iterable, Callable, Any, Optional, Mapping, TypeVar, Union
from autograd.numpy import ndarray

# default values
_default_parameters = {'rho': 1025.0, 'g': 9.81, 'depth': np.infty}
_default_min_damping = 1e-6

TWEC = TypeVar("TWEC", bound="WEC")
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

class NonlinearInvertedPendulumPTO:
    """A nonlinear inverted pendulum power take-off (PTO) object to be used 
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
                spring_stiffness: Optional[float] = 955,
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
                generator_max_conti_current = 17.1,  #A     #max conti torque = 17.1*0.186 = 3.18Nm, on gearbox side 3.18*6.6 = 21Nm > than gear box limit
                generator_coulomb_friction: Optional[float] = 0.6125, #Nm #coulomb friction, in the space of the gear box output
                drivetrain_friction: Optional[float] = 0.2,
                drivetrain_stiffness: Optional[float] = 0,
                nsubsteps_constraints: Optional[int] = 4,
        )  -> None:
        pto_gear_ratio = belt_gear_ratio*gearbox_gear_ratio
        freqs = wot.frequency(f1, nfreq, False)
        omega = 2*np.pi*freqs
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
        self.nsubsteps_constraints = nsubsteps_constraints
        self.pto_impedance = self._pto_impedance(omega,
                        pto_gear_ratio = pto_gear_ratio,
                        torque_constant = generator_torque_constant,
                        drivetrain_inertia = generator_rotor_inertia+gearbox_moi,
                        drivetrain_stiffness = drivetrain_stiffness,
                        drivetrain_friction = drivetrain_friction,
                        winding_resistance = generator_winding_resistance,
                        winding_inductance = generator_winding_inductance)
        self.pto_transfer_mat = self._pto_transfer_mat(self.pto_impedance)
        # self.torque_from_PTO, self.nstate_pto, self.nstate_opt =  self._create_control_torque_function
        self.nstate_pto, self.nstate_opt = self.nstate_based_on_control()
        self.f_add = {
            'Generator': self.torque_from_PTO,
            'Friction': self.nonlinear_torque_from_friction,
            'Spring': self.torque_from_nl_spring,
            'Pendulum': self.torque_from_pendulum,
        }

        self.constraints = [
            {'type': 'eq', 'fun': self.pendulum_residual_nl_spring}, # pendulum EoM
            {'type': 'ineq', 'fun': self.constraint_max_generator_torque},
        ]

        #here everything that needs initialization
        #trasnfermat
        #torque_from_pto, or just torque..but multiple torques...
        #other properties that we'd want later...
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
    # def _create_control_torque_function(self):
    # def _create_control_torque_function(self, wec, x_wec, x_opt, waves, nsubsteps=1):
    #     control_type = self.control_type
    #     nfreq = self.nfreq
    #     nstate_pen = self.nstate_pen
    #     if control_type == 'unstructured':
    #         nstate_pto = 2 * nfreq
    #         nstate_opt = nstate_pto + nstate_pen

    #         def torque_from_PTO(self, wec, x_wec, x_opt, waves=None, nsubsteps=1):
    #             f_fd = np.reshape(x_opt[:self.nstate_pto], (-1, ndof), order='F')  # Take the first components for PTO torque
    #             time_matrix = wec.time_mat_nsubsteps(nsubsteps)
    #             torque = np.dot(time_matrix, f_fd)  
    #             return torque

    #     elif control_type == 'damping':
    #         nstate_pto = 1
    #         nstate_opt = nstate_pto + nstate_pen

    #         def torque_from_PTO(self, wec, x_wec, x_opt, waves=None, nsubsteps=1):
    #             pos_rel = self.x_rel(wec, x_wec, x_opt)
    #             vel_rel = np.dot(wec.derivative_mat, pos_rel)
    #             f_fd = x_opt[:self.nstate_pto] * vel_rel
    #             time_matrix = wec.time_mat_nsubsteps(nsubsteps)
    #             torque = np.dot(time_matrix, f_fd)  
    #             return torque
    #     elif control_type == 'PI':
    #         nstate_pto = 2
    #         nstate_opt = nstate_pto + nstate_pen
    #         def torque_PI(vel, pos, coeffs):
    #             return (coeffs[0] * vel +  
    #                 coeffs[1] * pos) 
    #         def torque_from_PTO(self, wec, x_wec, x_opt, waves=None, nsubsteps=1):
    #             pos_rel = self.x_rel(wec, x_wec, x_opt)
    #             vel_rel = np.dot(wec.derivative_mat, pos_rel)
    #             f_fd = torque_PI(vel_rel, pos_rel, x_opt[:self.nstate_pto])
    #             time_matrix = wec.time_mat_nsubsteps(nsubsteps)
    #             torque = np.dot(time_matrix, f_fd)
    #             return torque
    #     elif control_type == 'PID':
    #         nstate_pto = 3
    #         nstate_opt = nstate_pto + nstate_pen
    #         def torque_PID(vel, pos, acc, coeffs):
    #             return (coeffs[0] * vel +  
    #                 coeffs[1] * pos+
    #                 coeffs[2] * acc) 
    #         def torque_from_PTO(self, wec, x_wec, x_opt, waves=None, nsubsteps=1):
    #             pos_rel = self.x_rel(wec, x_wec, x_opt)
    #             vel_rel = np.dot(wec.derivative_mat, pos_rel)
    #             acc_rel = np.dot(wec.derivative_mat, vel_rel)
    #             f_fd = torque_PID(vel_rel, pos_rel, acc_rel, x_opt[:self.nstate_pto])
    #             time_matrix = wec.time_mat_nsubsteps(nsubsteps)
    #             torque = np.dot(time_matrix, f_fd)
    #             return torque
    #     elif control_type == 'nonlinear_3rdO':
    #         nstate_pto = 7
    #         nstate_opt = nstate_pto + nstate_pen
    #         def torque_3rd_polynomial(vel, pos, coeffs):
    #             return (coeffs[0] +  #e1
    #                 coeffs[1] * vel +  #e1
    #                 coeffs[2] * pos + #e1
    #                 coeffs[3] * vel**3 + #e1
    #                 coeffs[4] * pos**3 + #e1
    #                 coeffs[5] * vel**2 * pos + #e0
    #                 coeffs[6] * vel * pos**2) #e1
    #         def torque_from_PTO(self,wec, x_wec, x_opt, waves=None, nsubsteps=1):
    #             pos_rel = self.x_rel(wec, x_wec, x_opt)
    #             vel_rel = np.dot(wec.derivative_mat, pos_rel)
    #             f_fd = torque_3rd_polynomial(vel_rel, pos_rel, x_opt[:self.nstate_pto])
    #             time_matrix = wec.time_mat_nsubsteps(nsubsteps)
    #             torque = np.dot(time_matrix, f_fd) 
    #             return torque

    #     else:
    #         raise ValueError("Invalid control type. Choose 'unstructured', 'damping', 'PI', 'PID', or 'nonlinear_3rdO'.")

    #     return torque_from_PTO, nstate_pto, nstate_opt
    
    def x_rel(self, wec, x_wec, x_opt, waves, nsubsteps=1):
        x_pos_buoy = wec.vec_to_dofmat(x_wec)
        x_pos_pen = wec.vec_to_dofmat(x_opt[self.nstate_pto:])
        return x_pos_buoy - x_pos_pen

    def rel_position(self, wec, x_wec, x_opt, waves, nsubsteps=1):
        pos_rel = self.x_rel(wec, x_wec, x_opt, waves)
        time_matrix = wec.time_mat_nsubsteps(nsubsteps)
        return np.dot(time_matrix, pos_rel)

    def rel_velocity(self, wec, x_wec, x_opt, waves, nsubsteps=1):
        pos_rel = self.x_rel(wec, x_wec, x_opt, waves)
        vel_rel = np.dot(wec.derivative_mat, pos_rel)
        time_matrix = wec.time_mat_nsubsteps(nsubsteps)
        return np.dot(time_matrix, vel_rel)
    ## PTO torque depending on controller

    def torque_from_PTO(self, wec, x_wec, x_opt, waves=None, nsubsteps=1):
        # Call the appropriate torque calculation method based on the control type
        if self.control_type == 'unstructured':
            return self.torque_from_unstructured(wec, x_wec, x_opt, waves, nsubsteps)
        elif self.control_type == 'damping':
            return self.torque_from_damping(wec, x_wec, x_opt, waves, nsubsteps)
        elif self.control_type == 'PI':
            return self.torque_from_PI(wec, x_wec, x_opt, waves, nsubsteps)
        elif self.control_type == 'PID':
            return self.torque_from_PID(wec, x_wec, x_opt, waves, nsubsteps)
        elif self.control_type == 'nonlinear_3rdO':
            return self.torque_from_nonlinear_3rdO(wec, x_wec, x_opt, waves, nsubsteps)
        else:
            raise ValueError("Invalid control type. Choose 'unstructured', 'damping', 'PI', 'PID', or 'nonlinear_3rdO'.")

    def nstate_based_on_control(self):
        control_type = self.control_type
        nfreq = self.nfreq
        nstate_pen = self.nstate_pen
        if control_type == 'unstructured':
            nstate_pto = 2 * nfreq
            nstate_opt = nstate_pto + nstate_pen
            return nstate_pto, nstate_opt
        elif control_type == 'damping':
            nstate_pto = 1
            nstate_opt = nstate_pto + nstate_pen
            return nstate_pto, nstate_opt
        elif control_type == 'PI':
            nstate_pto = 2
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
            raise ValueError("Invalid control type. Choose 'unstructured', 'damping', 'PI', 'PID', or 'nonlinear_3rdO'.")

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

    def torque_PI(self, vel, pos, coeffs):
        return (coeffs[0] * vel +  
                coeffs[1] * pos)

    def torque_PID(self, vel, pos, acc, coeffs):
        return (coeffs[0] * vel +  
                coeffs[1] * pos +
                coeffs[2] * acc)

    def torque_3rd_polynomial(self, vel, pos, coeffs):
        return (coeffs[0] +  # e1
                coeffs[1] * vel +  # e1
                coeffs[2] * pos +  # e1
                coeffs[3] * vel**3 +  # e1
                coeffs[4] * pos**3 +  # e1
                coeffs[5] * vel**2 * pos +  # e0
                coeffs[6] * vel * pos**2)  # e1

    ## additional torque

    def nonlinear_torque_from_friction(self, wec, x_wec, x_opt, waves, nsubsteps = 1):
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
        return  -self.spring_stiffness * scale * new_pos

    def torque_from_nl_spring(self, wec, x_wec, x_opt, waves, nsubsteps = 1):
        rel_pos = self.rel_position(wec, x_wec, x_opt, waves, nsubsteps) 
        spring_pos = self.spring_gear_ratio * rel_pos
        spring_torque = self.nonlinear_spring_torque(spring_pos)
        spring_torque_on_shaft = self.spring_gear_ratio * spring_torque
        return spring_torque_on_shaft

    def torque_from_pendulum(self, wec, x_wec, x_opt, waves, nsubsteps=1):
        pos_pen = wec.vec_to_dofmat(x_opt[self.nstate_pto:])
        time_matrix = wec.time_mat_nsubsteps(nsubsteps)
        pos_pen = np.dot(time_matrix, pos_pen)
        return -1*self.pendulum_mass * _default_parameters['g'] * self.pendulum_com * np.sin(pos_pen)
    ## residual

    def pendulum_inertia(self, wec, x_wec, x_opt, waves = None, nsubsteps = 1):
        pos_pen = wec.vec_to_dofmat(x_opt[self.nstate_pto:])
        acc_pen = np.dot(wec.derivative2_mat, pos_pen)
        time_matrix = wec.time_mat_nsubsteps(nsubsteps)
        acc_pen = np.dot(time_matrix, acc_pen)
        return self.pendulum_moi * acc_pen
    
    def pendulum_residual_nl_spring(self, wec, x_wec, x_opt, waves = None, nsubsteps = 1):
        resid = (
        self.pendulum_inertia(wec, x_wec, x_opt, waves, nsubsteps) +
        self.torque_from_pendulum(wec, x_wec, x_opt, waves, nsubsteps) +
        self.torque_from_nl_spring(wec, x_wec, x_opt, waves, nsubsteps) +
        self.nonlinear_torque_from_friction(wec, x_wec, x_opt, waves, nsubsteps) +
        self.torque_from_PTO(wec, x_wec, x_opt, waves, nsubsteps)
        )
        return resid.flatten()

    ## constraints
    
    #TODO: How to pass substeps?
    def constraint_max_generator_torque(self, wec, x_wec, x_opt, waves, nsubsteps = 5):
        torque = self.torque_from_PTO(wec, x_wec, x_opt, waves, nsubsteps)
        return self.max_PTO_torque - np.abs(torque.flatten())

    ## objective function
    
    def mechanical_power(self, wec, x_wec, x_opt, waves, nsubsteps=1):
        torque_td = self.torque_from_PTO(wec, x_wec, x_opt, waves, nsubsteps)
        vel_td = self.rel_velocity(wec, x_wec, x_opt, waves, nsubsteps)
        return vel_td * torque_td

    def electrical_power(self, wec, x_wec, x_opt, waves, nsubsteps=1):
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
        return q2_td * e2_td

    def energy(self, wec, x_wec, x_opt, waves, nsubsteps=1):
        power_td = self.electrical_power(wec, x_wec, x_opt, waves, nsubsteps)
        return np.sum(power_td) * wec.dt/nsubsteps

    def average_electrical_power(self, wec, x_wec, x_opt, waves, nsubsteps=1):
        e = self.energy(wec, x_wec, x_opt, waves, nsubsteps)
        return e / wec.tf

