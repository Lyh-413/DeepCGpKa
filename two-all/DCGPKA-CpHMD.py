from model import *
import scienceplots
rank = constants.DEVICE_NAME
def get_model():
    model = DeepCGpKa()
    model.load_state_dict({k.replace('module.', ''): v for k, v in
                           torch.load('2b-all.pkl').items()})
    model.to(rank)
    model.eval()
    return model
def eval_():
    model=get_model()
    test_dataset = CGDataSet('cphmddata')
    test_loader = DataLoaderX(dataset=test_dataset, batch_size=1)
    with torch.no_grad():
        pred_data = []
        label_data = []
        for i, (pdb_idx, seq, dismap, addseq, relpos, pos, label) in enumerate(test_loader):
            pred = model(seq.to(rank), dismap.to(rank), addseq.to(rank), relpos.to(rank), pos.to(rank)).squeeze()
            pred = pred.detach().cpu().numpy()
            pred_data.append(float(pred))
            label_data.append(label.squeeze().squeeze().detach().cpu().numpy())
    pred_data = np.array(pred_data)
    label_data = np.array(label_data)
    pearsonr = DataWindow.cal_pearsonr(self=None, label_arr=label_data,
                                       pred_arr=pred_data)
    DataWindow.test_scatter_plot(self=None, pred_list=pred_data, label_list=label_data, pearsonr=pearsonr)
eval_()
