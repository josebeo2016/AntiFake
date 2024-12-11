import torch
import numpy as np

# To load pretrained on vox2 model with Large-Margin finetuning:
model = torch.hub.load('IDRnD/ReDimNet', 'ReDimNet', model_name='b6', train_type='ft_lm', dataset='vox2')
model.cuda()
model.eval()

def asv_score(x: np.ndarray, y: np.ndarray) -> float:
    # x, y are numpy arrays with shape (n_samples, )
    global model

    # score is a float
    x = torch.from_numpy(x).unsqueeze(0).cuda().float()
    y = torch.from_numpy(y).unsqueeze(0).cuda().float()
    
    with torch.no_grad():
        score_x= model(x)
        score_y= model(y)
        # cosine similarity
        score = torch.nn.functional.cosine_similarity(score_x, score_y)
    return score.item()

if __name__ == '__main__':
    # test
    x = np.random.randn(16000)
    y = np.random.randn(16000)
    print(asv_score(x, x))
    print(asv_score(x, y))
