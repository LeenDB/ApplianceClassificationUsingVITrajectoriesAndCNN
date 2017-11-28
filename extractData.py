import os
import numpy as np

import cPickle as pickle

Data_path = '../../2016_07 - PLAID dataset/PLAID/'
csv_path = Data_path + 'CSV/'

import subprocess

def read_data_given_id(path,ids,progress=False,last_offset=0):
    '''read data given a list of ids and CSV paths'''
    n = len(ids)
    if n == 0:
        return {}
    else:
        data = {}
        for (i,ist_id) in enumerate(ids, start=1):
            if progress:
                print('%d/%d is being read...'%(i,n))
            if last_offset==0:
                data[ist_id] = np.genfromtxt(path+str(ist_id)+'.csv',
                delimiter=',',names='current,voltage',dtype=(float,float))
            else:
                p=subprocess.Popen(['tail','-'+str(int(offset)),path+
                    str(ist_id)+'.csv'],stdout=subprocess.PIPE)
                data[ist_id] = np.genfromtxt(p.stdout,delimiter=',',
                    names='current,voltage',dtype=(float,float))
        return data

def clean_meta(ist):
    '''remove '' elements in Meta Data '''
    clean_ist = ist.copy()
    for k,v in ist.items():
        if len(v) == 0:
            del clean_ist[k]
    return clean_ist

def parse_meta(meta):
    '''parse meta data for easy access'''
    M = {}
    for m in meta:
        for app in m:
            M[int(app['id'])] = clean_meta(app['meta'])
    return M

import json

def main(bins = 20, subsample = False):
    with open(Data_path + 'meta1.json') as data_file:
        meta1 = json.load(data_file)

    meta = [meta1]
    Meta = parse_meta(meta)

    # the locations
    Houses = [x['location'] for x in Meta.values()]
    # unique locations
    Unq_house = list(set(Houses))

    # appliance types of all instances
    Types = [x['type'] for x in Meta.values()]

    all_houses = {}
    all_traj = {}
    for bins in [20,30,40]:
        all_houses[bins] = {}
        all_traj[bins] = {}
    
    for id in range(1,len(Houses)+1):
    #for id in [1,100,200,400]:
        print(id)
        poss_data = read_data_given_id(csv_path, [id], False)
        I_inst = poss_data[id]['current']
        V_inst = poss_data[id]['voltage']

        if subsample:
            I_inst = I_inst[0:-1:6]
            V_inst = V_inst[0:-1:6]

        # collect the current and voltage
        v = {}
        v['current'] = I_inst
        v['voltage'] = V_inst

        # collect the trajectory
        v['current'][int(-1e4):] = v['current'][int(-1e4):] / max(abs(v['current'][int(-1e4):]))
        v['voltage'][int(-1e4):] = v['voltage'][int(-1e4):] / max(abs(v['voltage'][int(-1e4):]))

        
        for bins in [20,30,40]:
            
            xbins = np.linspace(-1, 1, num=bins+1)
            ybins = np.linspace(-1, 1, num = bins+1)

            xi = 0
            m = np.zeros((bins,bins))
            for x1, x2 in zip(xbins[:-1],xbins[1:]):
                yi = bins-1

                for y1, y2 in zip(ybins[:-1],ybins[1:]):
                    m[yi, xi] = sum((x1 <= v['current'][int(-1e4):]) & (v['current'][int(-1e4):] < x2) &
                                   (y1 <= v['voltage'][int(-1e4):]) & (v['voltage'][int(-1e4):] < y2))

                    yi -= 1
                xi += 1
            m = m / np.max(m)

            t = Types[id-1]
            h = Houses[id-1]
            if t not in all_houses[bins] and t not in all_traj[bins]:
                all_houses[bins][t] = []
                all_traj[bins][t] = []

            all_houses[bins][t].append(h)
            all_traj[bins][t].append(m)

    for nr in range(1,56):
    #for nr in range(1,3):
        for bins in [20,30,40]:
            t_data_test = {}
            t_data_training = {}
            house_testing = 'house' + str(nr)

            for appliance in all_houses[bins]:
                for i in range(len(all_houses[bins][appliance])):
                    print(i)
                    h = all_houses[bins][appliance][i]
                    traj = all_traj[bins][appliance][i]
                    if h == house_testing:
                        if appliance not in t_data_test:
                            t_data_test[appliance] = []
                        t_data_test[appliance].append(traj)
                    else:
                        if appliance not in t_data_training:
                            t_data_training[appliance] = []
                        t_data_training[appliance].append(traj)

            if not subsample:
                pickle.dump( t_data_test, open( "traj_"+str(bins)+"/test"+ str(nr) +"_"+str(bins) +"_traj.p", "wb" ) )
                pickle.dump( t_data_training, open( "traj_"+str(bins)+"/train"+ str(nr)+"_"+str(bins) +"_traj.p", "wb" ) )


if __name__ == "__main__":
    main(bins = 20)
