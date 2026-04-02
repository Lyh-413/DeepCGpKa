import numpy as np
import scipy.spatial
from biopandas.pdb import PandasPdb
import csv
from Bio.PDB import PDBIO, PDBParser
coor_cols = ['x_coord', 'y_coord', 'z_coord']
pka=[]
pdb_id=[]
res_id=[]
chain=[]
delres_id=[]
delpdb_id=[]
#use_num =30
with open(r"./val_n27.csv", encoding='utf-8')as f:
    reader = csv.reader(f)
    headers = next(reader)
    for row in reader:
        t=float(row[3])
        r=row[2]
        if r == 'ASP':
            t = t - 3.67
        if r == 'GLU':
            t = t - 4.25
        if r == 'HIS':
            t = t - 6.54
        if r == 'LYS':
            t = t - 10.40
        if abs(float(t))<=2.0:
            pka.append(t)
            pdb_id.append(row[0])
            res_id.append(row[1])
            chain.append(row[4])
#print(len(pka))
def getcos(x,y):
    left = np.linalg.norm(x)
    right = np.linalg.norm(y)
    middle = np.inner(x, y)
    if right == 0.:
        return  0.
    elif left == 0.:
        return 0.
    else:
        return middle / (left * right)
for v in range (len(pka)):
    '''
    sturcture=PDBParser().get_structure(id=None,file='./pdb2/%s.pdb'%(pdb_id[v]))
    io = PDBIO()
    sturcture=sturcture.get_chains()
    for c in sturcture:
        if c.get_id()==chain[v]:
            sturcture=c
            print(chain[v])
    io.set_structure(sturcture)
    io.save('out.pdb')
    '''
    pdb_data = PandasPdb().read_pdb('./pdb/%s_%s.pdb'%(pdb_id[v],chain[v])).df['ATOM']
    #pdb_data = PandasPdb().read_pdb('out.pdb').df['ATOM']
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
            #print('%sunneed'%(pdb_id[v]))
    if unneednum>0:
        for i in range(len(CAunneed)):
            #print(pdb_CA)
            index = pdb_CA[pdb_CA['line_idx'].values[int(CAunneed[len(CAunneed)-1-i])] == pdb_CA.line_idx].index.tolist()[0]
            #pdb_CB=pdb_CB.drop(index=index)
            pdb_CA=pdb_CA.drop(index=index)
    #print(resi_num[0])
    '''
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
    '''
    seq = pdb_CA['residue_name'].values

    dismap = scipy.spatial.distance.cdist(pdb_CA[coor_cols], pdb_CA[coor_cols], metric='euclidean')
    total_resi_num=len(seq)
    #if len(pdb_CA) > 500:
        #print('remove seq > 500: {}'.format(i))
    #else:
    label = pka[v]
    #res=int(res_id[v])-resi_num[0]
    '''
    if v <4871:
        idx = int(res)
    elif v >4913:
        idx=int(res)
    else:
        idx=int(res)-1
        '''
    idx = 0
    resi_num = pdb_CA['residue_number'].values
    resi_num=list(resi_num)
    #print(resi_num)
    for i in range(len(seq)):
        if int(resi_num[i])==int(res_id[v]):
            idx=i
    aaimvec=np.zeros((len(seq),3))
    '''
    abimvec=np.zeros((len(seq),3))
    baimvec = np.zeros((len(seq), 3))
    bbimvec = np.zeros((len(seq), 3))
    '''
    dislist=pdb_CA[coor_cols].to_numpy()
    #dislistb=pdb_CB[coor_cols].to_numpy()
    #print(pdb_CB['residue_number'].values)
    for i in range(len(seq)):
        for j in  range(3):
            aaimvec[i][j]=dislist[i][j]-dislist[idx][j]
            #baimvec[i][j]=dislistb[i][j]-dislist[idx][j]
            #abimvec[i][j]=dislist[i][j]-dislistb[idx][j]
            #bbimvec[i][j]=dislistb[i][j]-dislistb[idx][j]
    foucsdis = dismap[idx]
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
    dismapneed=np.zeros((use_num,use_num,3))
    seqi=[]
    #cthetaneed=np.zeros((use_num,use_num,4))
    #squareneed=np.zeros((use_num,use_num))
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
            dismapneed[i][j][0]=np.linalg.norm(aaimvec[int(noofres[i])])
            #dismapneed[i][j][1] = np.linalg.norm(abimvec[int(noofres[i])])
            #dismapneed[i][j][2] = np.linalg.norm(baimvec[int(noofres[i])])
            #dismapneed[i][j][3] = np.linalg.norm(bbimvec[int(noofres[i])])
            dismapneed[i][j][1] = np.linalg.norm(aaimvec[int(noofres[j])])
            #dismapneed[i][j][5] = np.linalg.norm(abimvec[int(noofres[j])])
            #dismapneed[i][j][6] = np.linalg.norm(baimvec[int(noofres[j])])
            #dismapneed[i][j][7] = np.linalg.norm(bbimvec[int(noofres[j])])
            dismapneed[i][j][2] = np.linalg.norm(np.subtract(aaimvec[int(noofres[i])],aaimvec[int(noofres[j])]))
            #dismapneed[i][j][9] = np.linalg.norm(np.subtract(aaimvec[int(noofres[i])],baimvec[int(noofres[j])]))
            #dismapneed[i][j][10] = np.linalg.norm(np.subtract(baimvec[int(noofres[i])],aaimvec[int(noofres[j])]))
            #dismapneed[i][j][11] = np.linalg.norm(np.subtract(baimvec[int(noofres[i])],baimvec[int(noofres[j])]))
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
    np.savez('./cphmddata' + '/{}.npz'.format(v),
                 seq=seqi,
                 dismap=dismapneed,
                 #square=squareneed,
                 #ctheta=cthetaneed,
                 label=label,
                 relpos=noofres,
                 res_id=idx)
