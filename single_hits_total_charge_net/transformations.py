"""Transformation tensorflow layers"""
import tensorflow as tf
from scipy import constants


class trafo_indep(tf.keras.layers.Layer):
    '''Class to transfor inputs for Hit Net
    Independent angles
    '''
    speed_of_light = constants.c * 1e-9 # c in m / ns

    
    def __init__(self, labels):
        
        super().__init__()

        self.labels = labels
        
        self.azimuth_idx = labels.index('azimuth')
        self.zenith_idx = labels.index('zenith')
        self.time_idx = labels.index('time')
        self.x_idx = labels.index('x')
        self.y_idx = labels.index('y')
        self.z_idx = labels.index('z')
        self.cascade_energy_idx = labels.index('cascade_energy')
        self.track_energy_idx = labels.index('track_energy')
        
    def get_config(self):
        return {'labels': self.labels}

    def call(self, hit, params):
        
        cosphi = tf.math.cos(params[:, self.azimuth_idx])
        sinphi = tf.math.sin(params[:, self.azimuth_idx])
        
        sintheta = tf.math.sin(params[:, self.zenith_idx])
        dir_x = sintheta * cosphi
        dir_y = sintheta * sinphi
        dir_z = tf.math.cos(params[:, self.zenith_idx])
        
        dx = params[:, self.x_idx] - hit[:,0]
        dy = params[:, self.y_idx] - hit[:,1]
        dz = params[:, self.z_idx] - hit[:,2]
        
        
        rho = tf.math.sqrt(tf.math.square(dx) + tf.math.square(dy))
        dist = tf.math.sqrt(tf.math.square(dx) + tf.math.square(dy) + tf.math.square(dz))     
        
        absdeltaphidir = tf.abs(tf.math.acos(
                                tf.clip_by_value(-tf.math.divide_no_nan((cosphi*dx + sinphi*dy), rho),
                                                 clip_value_min = -1.,
                                                 clip_value_max = +1.,
                                                )
                                            )
                                )
        costheta = tf.math.divide_no_nan(rho, dist)

        dt = hit[:,3] - params[:, self.time_idx]
              
        # difference c*t - r
        delta = dt * self.speed_of_light - dist

        cascade_energy = params[:, self.cascade_energy_idx]
        track_energy = params[:, self.track_energy_idx]
        
        out = tf.stack([
                 delta,
                 dist,
                 costheta,
                 absdeltaphidir,
                 dir_x,
                 dir_y,
                 dir_z,
                 dx,
                 dy,
                 dz,
                 params[:, self.x_idx],
                 params[:, self.y_idx],
                 params[:, self.z_idx],
                 cascade_energy,
                 track_energy
                ],
                axis=1
                )    
            
        return out
    
    
class trafo_q(tf.keras.layers.Layer):
    '''Class to transfor inputs for Charget Net
    '''
    
    def __init__(self, labels):
        
        super().__init__()
        
        self.labels = labels
        
        self.azimuth_idx = labels.index('azimuth')
        self.zenith_idx = labels.index('zenith')
        self.x_idx = labels.index('x')
        self.y_idx = labels.index('y')
        self.z_idx = labels.index('z')
        self.cascade_energy_idx = labels.index('cascade_energy')
        self.track_energy_idx = labels.index('track_energy')
   
    def get_config(self):
        return {'labels': self.labels}
    
    def call(self, charge, params):
        
        dir_x = tf.math.sin(params[:, self.zenith_idx]) * tf.math.cos(params[:, self.azimuth_idx])
        dir_y = tf.math.sin(params[:, self.zenith_idx]) * tf.math.sin(params[:, self.azimuth_idx])
        dir_z = tf.math.cos(params[:, self.zenith_idx])

        out = tf.stack([
                 charge[:,0],
                 params[:, self.x_idx],
                 params[:, self.y_idx],
                 params[:, self.z_idx],
                 dir_x,
                 dir_y,
                 dir_z,
                 params[:, self.cascade_energy_idx],
                  params[:, self.track_energy_idx]
                ],
                axis=1
                )            

        return out