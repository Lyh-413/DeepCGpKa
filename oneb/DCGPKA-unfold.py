from unfold import *
from model import *
def aipredpka(s):
    x='%s.pdb'%(s)
    pkadatacaculate(x)
    model=DeepCGpKa()
    model.load_state_dict({k.replace('module.', ''): v for k, v in
                           torch.load('1b.pkl').items()})
    rank='cuda'
    model.to(rank)
    model.eval()
    test_dataset = CGDataSet('unfolddata')
    test_loader = DataLoaderX(dataset=test_dataset, batch_size=1)
    with torch.no_grad():
        pred_data = np.zeros(175)
        label_data = []
        r_data = []
        pdb_name = []
        for i, (pdb_idx, seq, dismap, addseq, relpos, pos, label) in enumerate(test_loader):
            pred = model(seq.to(rank), dismap.to(rank), addseq.to(rank), relpos.to(rank), pos.to(rank)).squeeze()
            '''
            if addseq[0][0]==1:
                pred = pred.detach().cpu().numpy()+3.67
            if addseq[0][1]==1:
                pred = pred.detach().cpu().numpy()+4.25
            if addseq[0][2] == 1:
                pred = pred.detach().cpu().numpy() + 6.54
            if addseq[0][3]==1:
                pred = pred.detach().cpu().numpy()+10.40
            '''
            pred = pred.detach().cpu().numpy()
            #print("pred=%f,pdb=%s" % (pred, pdb_idx[0]), flush=True)

            pred_data[int(pdb_idx[0])]=float(pred)
            #label_data.append(label)
                # pred_data=test_dataset.detrans_label(pred_data)
        pred_data = np.array(pred_data)
        return pred_data
        #np.savez('pka.npz',pka=pred_data)
out=aipredpka('outt')
for i in range(5):
    out=np.concatenate((out,aipredpka('outt%d'%(i+1))))
np.savetxt('array_numpy.csv', out, delimiter=',')