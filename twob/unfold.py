import numpy as np
import scipy.spatial
from biopandas.pdb import PandasPdb
from Bio.PDB import PDBIO, PDBParser
from pandas.core.frame import DataFrame
def getpredres(x):
    resi_num = x['residue_number'].values
    aim=x[(x['residue_name']=='ASP') | (x['residue_name']=='GLU') | (x['residue_name']=='LYS') | (x['residue_name']=='HIS')]
    res_id=aim['residue_number'].values
    return res_id
def pkadatacaculate(x):
    coor_cols = ['x_coord', 'y_coord', 'z_coord']
    sturcture=PDBParser().get_structure(id='None',file=x)
    io = PDBIO()
    sturcture=sturcture.get_chains()
    number=int(0)
    for c in sturcture:
        io.set_structure(c)
        io.save('out.pdb')

    #pdb_data = PandasPdb().read_pdb('./pdb/%s_%s.pdb'%(pdb_id[v],chain[v])).df['ATOM']
        pdb_data = PandasPdb().read_pdb('out.pdb').df['ATOM']
    #aa_dismap = scipy.spatial.distance.cdist(pdb_data[coor_cols], pdb_data[coor_cols], metric='euclidean')
        pdb_CA = pdb_data[pdb_data["atom_name"] == 'CA']
        resi_num = pdb_CA['residue_number'].values
        total_resi_num = len(pdb_CA['residue_name'].values)
        CAunneed=[]
        unneednum=0
        for i in range(total_resi_num-1):
            if resi_num[i]==resi_num[i+1]:
                CAunneed.append(i+1)
                unneednum=unneednum+1
                print('%sunneed'%(i+1))
        if unneednum>0:
            for i in range(len(CAunneed)):
                #print(pdb_CA)
                index = pdb_CA[pdb_CA['line_idx'].values[int(CAunneed[len(CAunneed)-1-i])] == pdb_CA.line_idx].index.tolist()[0]
                #pdb_CB=pdb_CB.drop(index=index)
                pdb_CA=pdb_CA.drop(index=index)
    #print(resi_num[0])
        CBnum=[]
        for i in range(len(pdb_CA['residue_name'].values)):
            resatom= pdb_data[pdb_data['residue_number'] == pdb_CA['residue_number'].values[i]]
            if pdb_CA['residue_name'].values[i]=='GLY':
                dt=resatom[resatom['atom_name'] == 'C']
                if dt.shape[0]!=1:
                    CBnum.append(dt[dt['atom_number']==dt['atom_number'].values[0]])
                else:
                    CBnum.append(dt)
            else:
                dt=resatom[resatom['atom_name'] == 'CB']
                if dt.shape[0] != 1:
                    if dt.shape[0]==2:
                        CBnum.append(dt[dt['atom_number'] == dt['atom_number'].values[0]])
                    else:
                        CBnum.append(pdb_CA[pdb_CA['residue_number']==pdb_CA['residue_number'].values[i]])
                else:
                    CBnum.append(dt)
        CBnum=np.array(CBnum)
        #print(CBnum.shape)
        first=CBnum.shape[0]
        last=CBnum.shape[-1]
        CBnum=np.reshape(CBnum,(first,last))
        pdb_CB = DataFrame(CBnum,columns=pdb_CA.columns.values.tolist())
        CBcoorcols=pdb_CB[coor_cols].astype('float')
        seq = pdb_CA['residue_name'].values

        dismapaa = scipy.spatial.distance.cdist(pdb_CA[coor_cols], pdb_CA[coor_cols], metric='euclidean')
        dismapbb = scipy.spatial.distance.cdist(CBcoorcols, CBcoorcols, metric='euclidean')
        dismapab = scipy.spatial.distance.cdist(pdb_CA[coor_cols], CBcoorcols, metric='euclidean')
        dismapba = scipy.spatial.distance.cdist(CBcoorcols, pdb_CA[coor_cols], metric='euclidean')
        res_id=getpredres(pdb_CA)
        total_resi_num=len(seq)
        resi_num = pdb_CA['residue_number'].values
        resi_num=list(resi_num)
        #print(resi_num)
        for v in range( len(res_id) ):
            idx = 0
            for i in range(len(seq)):
                if int(resi_num[i]) == int(res_id[v]):
                    idx = i
            foucsdis = dismapaa[idx]
            #if total_resi_num > use_num or total_resi_num==use_num:
                #arr_min = heapq.nsmallest(use_num, foucsdis)
                #index_min=np.zeros(use_num)
                #for i in range(use_num):
                   # index_min[i] =np.where(foucsdis==arr_min[i])[0]
            noofres=[]
            #score=0
            for i in range(len(seq)):
                    if foucsdis[i]<=18.0:
                        noofres.append(int(i))
            use_num=len(noofres)
            '''
            aaimvec=np.zeros((use_num,3))
            abimvec=np.zeros((use_num,3))
            baimvec = np.zeros((use_num, 3))
            bbimvec = np.zeros((use_num, 3))
            dismapneed=np.zeros((use_num,use_num,12))
            seqi=[]
            #cthetaneed=np.zeros((use_num,use_num,4))
            #squareneed=np.zeros((use_num,use_num))
            for i in range(use_num):
                for j in  range(3):
                    aaimvec[i][j]=dislist[int(noofres[i])][j]-dislist[idx][j]
                    baimvec[i][j]=dislistb[int(noofres[i])][j]-dislist[idx][j]
                    abimvec[i][j]=dislist[int(noofres[i])][j]-dislistb[idx][j]
                    bbimvec[i][j]=dislistb[int(noofres[i])][j]-dislistb[idx][j]
            foucsdis = dismapaa[idx]
            '''
            dismapneed = np.zeros((use_num, use_num, 12))
            seqi = []
            for i in  range(use_num):
                seqi.append(seq[int(noofres[i])])
                for j in range(use_num):
                    '''
                    left=np.linalg.norm(aimvec[int(noofres[i])])
                    right=np.linalg.norm(aimvec[int(noofres[j])])
                    cthetaneed[i][j][0]=getcos(aimvec[int(noofres[i])],aimvec[int(noofres[j])])
                    cthetaneed[i][j][1]=getcos(selfvec[int(noofres[i])],selfvec[int(noofres[j])])
                    cthetaneed[i][j][2]=getcos(selfvec[int(noofres[i])],selfvec[idx])
                    cthetaneed[i][j][3]=getcos(selfvec[idx],selfvec[int(noofres[j])])
                    dismapneed[i][j][0]=left#dismap[int(noofres[i])][int(noofres[j])]
                    dismapneed[i][j][1]=right
                    #cro=np.cross(aimvec[int(noofres[i])],aimvec[int(noofres[j])])
                    #squareneed[i][j]=np.linalg.norm(cro)**0.5
                    if right==0.:
                        cthetaneed[i][j]=0.
                    elif left==0.:
                        cthetaneed[i][j] = 0.
                    else:
                        cthetaneed[i][j]=middle/(left*right)
                    '''
                    dismapneed[i][j][0] = dismapaa[int(noofres[i])][idx]
                    dismapneed[i][j][1] = dismapab[int(noofres[i])][idx]
                    dismapneed[i][j][2] = dismapba[int(noofres[i])][idx]
                    dismapneed[i][j][3] = dismapbb[int(noofres[i])][idx]
                    dismapneed[i][j][4] = dismapaa[int(noofres[j])][idx]
                    dismapneed[i][j][5] = dismapaa[int(noofres[j])][idx]
                    dismapneed[i][j][6] = dismapaa[int(noofres[j])][idx]
                    dismapneed[i][j][7] = dismapaa[int(noofres[j])][idx]
                    dismapneed[i][j][8] = dismapaa[int(noofres[i])][int(noofres[j])]
                    dismapneed[i][j][9] = dismapab[int(noofres[i])][int(noofres[j])]
                    dismapneed[i][j][10] = dismapba[int(noofres[i])][int(noofres[j])]
                    dismapneed[i][j][11] = dismapbb[int(noofres[i])][int(noofres[j])]
                    '''
            if total_resi_num<use_num:
                dismapneed=np.zeros((total_resi_num,total_resi_num))
                seqi=[]
                noofres = np.zeros(total_resi_num)
                cthetaneed=np.zeros((total_resi_num,total_resi_num))
                squareneed=np.zeros((total_resi_num,total_resi_num))
                for i in range(total_resi_num):
                    noofres[i]=i
                    seqi.append(seq[i])
                    for j in range(total_resi_num):
                        left = np.linalg.norm(aimvec[i])
                        right = np.linalg.norm(aimvec[j])
                        middle = np.inner(aimvec[i], aimvec[j])
                        dismapneed[i][j] = dismap[i][j]
                        cro = np.cross(aimvec[i], aimvec[j])
                        squareneed[i][j] = np.linalg.norm(cro)
                        if right == 0.:
                            cthetaneed[i][j]=0.
                        elif left == 0.:
                            cthetaneed[i][j]=0.
                        else:
                            cthetaneed[i][j]=middle/(left*right)
        '''
            np.savez('./unfolddata' + '/{}.npz'.format(v+number),
                         seq=seqi,
                         dismap=dismapneed,
                         #square=squareneed,
                         #ctheta=cthetaneed,
                         label=0.0,
                         relpos=noofres,
                         res_id=idx)
        number=number+len(res_id)
        np.savez('./outchainandresid%s.npz'%(c.get_id()),chain=c.get_id(),res_id=res_id)