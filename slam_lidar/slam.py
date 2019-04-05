from __future__ import division
import numpy as np
import slam_utils
import tree_extraction
def motion_model(u, dt, ekf_state, vehicle_params):
    '''
    Computes the discretized motion model for the given vehicle as well as its Jacobian

    Returns:
        f(x,u), a 3x1 vector corresponding to motion x_{t+1} - x_t given the odometry u.

        df/dX, the 3x3 Jacobian of f with respect to the vehicle state (x, y, phi)
    '''

    ###
    # Implement the vehicle model and its Jacobian you derived.
    ###
    a=vehicle_params['a']
    b=vehicle_params['b']
    H=vehicle_params['H']
    L=vehicle_params['L']
    ve=u[0]
    alpha=u[1]
    vc=ve/(1-np.tan(alpha)*H/L)
    motion=ekf_state['x']

    xv=ekf_state['x'][0]+dt*(vc*np.cos(ekf_state['x'][2])-vc/L*np.tan(alpha)*(a*np.sin(ekf_state['x'][2])+b*np.cos(ekf_state['x'][2])))
    yv=ekf_state['x'][1]+dt*(vc*np.sin(ekf_state['x'][2])+vc/L*np.tan(alpha)*(a*np.cos(ekf_state['x'][2])-b*np.sin(ekf_state['x'][2])))
    theta=slam_utils.clamp_angle(ekf_state['x'][2]+dt*vc/L*np.tan(alpha))
    motion[:3]=[xv,yv,theta]
    G = np.zeros([3, ekf_state['x'].shape[0]])
    xv_G=[1,0,dt*(-vc*np.sin(ekf_state['x'][2])-vc/L*np.tan(alpha)*(a*np.cos(ekf_state['x'][2])-b*np.sin(ekf_state['x'][2])))]
    yv_G=[0,1,dt*(vc*np.cos(ekf_state['x'][2])+vc/L*np.tan(alpha)*(-a*np.sin(ekf_state['x'][2])-b*np.cos(ekf_state['x'][2])))]
    theta_G=[0,0,1]
    G[:3,:3]=np.vstack([xv_G,yv_G,theta_G])


    return motion, G

def odom_predict(u, dt, ekf_state, vehicle_params, sigmas):
    '''
    Perform the propagation step of the EKF filter given an odometry measurement u 
    and time step dt where u = (ve, alpha) as shown in the vehicle/motion model.

    Returns the new ekf_state.
    '''

    ###
    # Implement the propagation
    ###
    motion ,G =motion_model(u, dt, ekf_state, vehicle_params)
    cor_prev=ekf_state['P']
    R=np.diag([sigmas['xy']**2,sigmas['xy']**2,sigmas['phi']**2])
    cor=G.dot(cor_prev).dot(G.T)+R
    cor=slam_utils.make_symmetric(cor)
    ekf_state['P'][:3,:3]=cor
    ekf_state['x']=motion

    return ekf_state


def gps_update(gps, ekf_state, sigmas):
    '''
    Perform a measurement update of the EKF state given a GPS measurement (x,y).

    Returns the updated ekf_state.
    '''
    
    ###
    # Implement the GPS update.
    ###

    P=ekf_state['P'].copy()
    x=ekf_state['x'].copy()
    H=np.zeros([2,x.shape[0]])
    H[:2,:2]=np.array([[1,0],[0,1]])
    Q=np.diag([sigmas['gps']**2,sigmas['gps']**2])
    S=H.dot(P).dot(H.T)+Q.T
    K=P.dot(H.T).dot(slam_utils.invert_2x2_matrix(S))
    r=(gps-x[:2]).reshape([2,1])
    if r.T.dot(slam_utils.invert_2x2_matrix(S)).dot(r) < 13.8:
        x_new=x+K.dot(gps-x[:2])
        cor_new=(np.eye(x.shape[0])-K.dot(H)).dot(P)
        cor_new=slam_utils.make_symmetric(cor_new)
        ekf_state['P']=cor_new
        ekf_state['x']=x_new

    # ekf_state=slam.gps_update(gps, ekf_state, sigmas)

    return ekf_state

def laser_measurement_model(ekf_state, landmark_id):
    ''' 
    Returns the measurement model for a (range,bearing) sensor observing the
    mapped landmark with id 'landmark_id' along with its jacobian. 

    Returns:
        h(x, l_id): the 2x1 predicted measurement vector [r_hat, theta_hat].
        dh/dX: For a measurement state with m mapped landmarks, i.e. a state vector of
                dimension 3 + 2*m, this should return the full 2 by 3+2m Jacobian
                matrix corresponding to a measurement of the landmark_id'th feature.
    '''
    
    ###
    # Implement the measurement model and its Jacobian you derived
    ###
    x=ekf_state['x'].copy()
    H = np.zeros([2,x.shape[0]])

    xl=x[3+landmark_id*2]
    yl=x[4+landmark_id*2]
    ssqrt = np.sqrt((x[0] - xl)**2 + (x[1] - yl)**2)
    H[0,:3]=[-(xl-x[0])/ssqrt, -(yl-x[1])/ssqrt, 0]

    H[0,3+landmark_id*2:5+landmark_id*2]=[(xl-x[0])/ssqrt, (yl-x[1])/ssqrt]
    ppow=((yl-x[1])/(xl-x[0]))**2+1
    H[1,:3]= [1/ppow*(yl-x[1])/(xl-x[0])**2, 1/ppow*(-1/(xl-x[0])),-1]
    H[1,3+landmark_id*2:5+landmark_id*2]=[-1/ppow*(yl-x[1])/(xl-x[0])**2, -1/ppow*(-1/(xl-x[0]))]
    zb=np.arctan2(yl-x[1],xl-x[0])-x[2]
    zhat=np.array([ssqrt,slam_utils.clamp_angle(zb)])

    # zhat, H=slam.laser_measurement_model(ekf_state, landmark_id)

    return zhat, H

def initialize_landmark(ekf_state, tree):
    '''
    Initialize a newly observed landmark in the filter state, increasing its
    dimension by 2.

    Returns the new ekf_state.
    '''

    ###
    # Implement this function.
    ###
    tree=np.reshape(tree,[-1,3])
    tree_xy = np.array(slam_utils.tree_to_global_xy(tree, ekf_state))

    if tree_xy.shape[0]>0:
        num=tree.shape[0]
        x=ekf_state['x'].copy()
        cor_update=np.eye(num*2+x.shape[0])*1000
        cor_update[:x.shape[0],:x.shape[0]]=ekf_state['P']
        ekf_state['P']=cor_update
        ekf_state['x']=np.append(x,tree_xy.flatten('F'))
        ekf_state['num_landmarks']+=num

    # ekf_state=slam.initialize_landmark(ekf_state, tree)

    return ekf_state

def compute_data_association(ekf_state, measurements, sigmas, params):
    '''
    Computes measurement data association.

    Given a robot and map state and a set of (range,bearing) measurements,
    this function should compute a good data association, or a mapping from 
    measurements to landmarks.

    Returns an array 'assoc' such that:
        assoc[i] == j if measurement i is determined to be an observation of landmark j,
        assoc[i] == -1 if measurement i is determined to be a new, previously unseen landmark, or,
        assoc[i] == -2 if measurement i is too ambiguous to use and should be discarded.
    '''

    if ekf_state["num_landmarks"] == 0:
        # set association to init new landmarks for all measurements
        return [-1 for m in measurements]

    ###
    # Implement this function.
    ###
    cor=ekf_state['P']
    num_obs=len(measurements)
    measurements = np.array(measurements)[:,:2]
    assoc = np.full([num_obs,],-2)
    num_mark=ekf_state['num_landmarks']
    M=np.zeros([num_obs,num_mark])
    Q = np.diag([sigmas['range']**2, sigmas['bearing']**2])
    for i in range(num_mark):

        zhat, H = laser_measurement_model(ekf_state, i)
        S=H.dot(cor).dot(H.T)+Q.T
        for j in range(num_obs):
            r=(measurements[j,:]-zhat).reshape([2,1])
            M[j,i]=r.T.dot(slam_utils.invert_2x2_matrix(S)).dot(r)
    if num_mark>=num_obs:
        pos = np.array(slam_utils.solve_cost_matrix_heuristic(M.copy()))
        posx = pos[:, 0]
        posy = pos[:, 1]
        value = M[posx, posy]
        ind_update = np.where(value >= 9.2103)
        ind_discard = np.where((value > 5.9915) & (value < 9.2103))
        pos[ind_update, 1] = -1
        pos[ind_discard, 1] = -2
        assoc[pos[:, 0]] = pos[:, 1]
    else:
        A1=np.full([num_obs,num_obs],5.99)
        A2 = np.full([num_obs, num_obs], 9.21)

        M_exp1=np.concatenate([M,A1],axis=1)
        M_exp2 = np.concatenate([M, A2], axis=1)
        pos1=np.array(slam_utils.solve_cost_matrix_heuristic(M_exp1.copy()))
        pos2 = np.array(slam_utils.solve_cost_matrix_heuristic(M_exp2.copy()))
        posx1=pos1[:,0]
        posy1=pos1[:,1]
        posx2 = pos2[:, 0]
        posy2 = pos2[:, 1]
        assoc[posx1[posy1<num_mark]]=posy1[posy1<num_mark]
        assoc[posx2[posy2 >= num_mark]] = -1

    # assoc=slam.compute_data_association(ekf_state, measurements, sigmas, params)

    return assoc

def laser_update(trees, assoc, ekf_state, sigmas, params):
    '''
    Perform a measurement update of the EKF state given a set of tree measurements.

    trees is a list of measurements, where each measurement is a tuple (range, bearing, diameter).

    assoc is the data association for the given set of trees, i.e. trees[i] is an observation of the
    ith landmark. If assoc[i] == -1, initialize a new landmark with the function initialize_landmark
    in the state for measurement i. If assoc[i] == -2, discard the measurement as 
    it is too ambiguous to use.

    The diameter component of the measurement can be discarded.

    Returns the ekf_state.
    '''

    ###
    # Implement the EKF update for a set of range, bearing measurements.
    ###
    if np.array(assoc).size == 0:
        return ekf_state


    assoc=np.array(assoc,dtype = np.int)
    dtrees=np.array(trees)[:,:2]
    new_ind=np.where(assoc==-1)[0]
    new_trees=dtrees[new_ind,:]

    prevnum=ekf_state['num_landmarks']
    ekf_state = initialize_landmark(ekf_state, np.array(trees)[new_ind, :])
    newnum=ekf_state['num_landmarks']

    update_ind=np.where(assoc>=0)[0]
    ekf_state_prev=ekf_state.copy()
    cor_prev=ekf_state_prev['P']
    x_prev=ekf_state_prev['x']



    for i in range(prevnum,newnum):
        z=new_trees[i-prevnum,:]
        zhat,H=laser_measurement_model(ekf_state_prev, i)
        Q = np.diag([sigmas['range']**2, sigmas['bearing']**2])
        K=cor_prev.dot(H.T).dot(slam_utils.invert_2x2_matrix(H.dot(cor_prev).dot(H.T)+Q.T))
        ekf_state['x'] +=K.dot(z-zhat)
        ekf_state['P'] +=-K.dot(H).dot(cor_prev)
    for i in update_ind[::-1]:
        ii=assoc[i]
        z=dtrees[i,:]
        zhat,H=laser_measurement_model(ekf_state_prev, ii)
        Q = np.diag([sigmas['range']**2, sigmas['bearing']**2])
        K=cor_prev.dot(H.T).dot(slam_utils.invert_2x2_matrix(H.dot(cor_prev).dot(H.T)+Q.T))
        ekf_state['x'] +=K.dot(z-zhat)
        ekf_state['P'] +=-K.dot(H).dot(cor_prev)







    return ekf_state


def run_ekf_slam(events, ekf_state_0, vehicle_params, filter_params, sigmas):
    last_odom_t = -1
    ekf_state = {
        'x': ekf_state_0['x'].copy(),
        'P': ekf_state_0['P'].copy(),
        'num_landmarks': ekf_state_0['num_landmarks']
    }
    
    state_history = {
        't': [0],
        'x': ekf_state['x'],
        'P': np.diag(ekf_state['P'])
    }

    if filter_params["do_plot"]:
        plot = slam_utils.init_plot()

    for i, event in enumerate(events):
        t = event[1][0]
        if i % 1000 == 0:
            print("t = {}".format(t))


        if event[0] == 'gps':
            gps_msmt = event[1][1:]
            ekf_state = gps_update(gps_msmt, ekf_state, sigmas)


        elif event[0] == 'odo':
            if last_odom_t < 0:
                last_odom_t = t
                continue
            u = event[1][1:]
            dt = t - last_odom_t
            ekf_state = odom_predict(u, dt, ekf_state, vehicle_params, sigmas)
            last_odom_t = t

        else:
            # Laser


            scan = event[1][1:]
            trees = tree_extraction.extract_trees(scan, filter_params)
            assoc = compute_data_association(ekf_state, trees, sigmas, filter_params)

            ekf_state = laser_update(trees, assoc, ekf_state, sigmas, filter_params)
            if filter_params["do_plot"]:
                slam_utils.do_plot(state_history['x'], ekf_state, trees, scan, assoc, plot, filter_params)


        state_history['x'] = np.vstack((state_history['x'], ekf_state['x'][0:3]))
        state_history['P'] = np.vstack((state_history['P'], np.diag(ekf_state['P'][:3,:3])))
        state_history['t'].append(t)

    return state_history


def main():
    odo = slam_utils.read_data_file("data/DRS.txt")
    gps = slam_utils.read_data_file("data/GPS.txt")
    laser = slam_utils.read_data_file("data/LASER.txt")

    # collect all events and sort by time
    events = [('gps', x) for x in gps]
    events.extend([('laser', x) for x in laser])
    events.extend([('odo', x) for x in odo])

    events = sorted(events, key = lambda event: event[1][0])


    vehicle_params = {
        "a": 3.78,
        "b": 0.50, 
        "L": 2.83,
        "H": 0.76
    }

    filter_params = {
        # measurement params
        "max_laser_range": 75, # meters

        # general...
        "do_plot": True,
        "plot_raw_laser": True,
        "plot_map_covariances": True

        # Add other parameters here if you need to...
    }

    # Noise values
    sigmas = {
        # Motion model noise
        "xy": 0.05,
        "phi": 0.5*np.pi/180,

        # Measurement noise
        "gps": 3,
        "range": 0.4,
        "bearing": 3*np.pi/180
    }

    # Initial filter state
    ekf_state = {
        "x": np.array( [gps[0,1], gps[0,2], 36*np.pi/180]),
        "P": np.diag([.1, .1, 1]),
        "num_landmarks": 0
    }

    run_ekf_slam(events, ekf_state, vehicle_params, filter_params, sigmas)

if __name__ == '__main__':
    main()
