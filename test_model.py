from calendar import c
from text2tensor import Text2Tensor
from sentiment_analysis import SentimentAnalysisNN
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f' ____Using {device}____')


def predition_step(
    model: torch.nn.Module,
    data: torch.tensor,
    device: torch.device,
    ):

    hidden = model.init_hidden(1) # batch size

    model.eval()
    with torch.inference_mode():
        #print(X.shape)
        #hidden = tuple([each.data for each in hidden])

        X = data.to(device).unsqueeze(dim=0)
        validation_pred, hidden = model(X, hidden)

        print(validation_pred)
    return torch.round(validation_pred).long().item()

vocab_size = 5000
vocab_size += 1 # novo id (0) para palavras não existentes no vocabulário

embedding_dim = 50
hidden_size_lstm = 64
num_layers_lstm = 2
output_shape = 1


model = SentimentAnalysisNN(
    device=device,
    input_shape=vocab_size,
    output_shape=output_shape,
    embedding_dim=embedding_dim,
    hidden_size_lstm=hidden_size_lstm,
).to(device)

state_dict = torch.load('data\models\SentimentAnalysisNN.pt')['model_state_dict']
model.load_state_dict(state_dict)

t2t = Text2Tensor(
    dict_path=f'data\ptbr\ptbr_imdb_{vocab_size-1}.json',
    dataset_path='data\imdb\imdb-reviews-pt-br.csv',
    X_col='text_pt',
    y_col='sentiment',
    n_text_inputs=1,
    max_text_size=256
)

def response(text):
    #text = input('Text: ')

    text_tensor, _ = t2t.text2tensor(text, 'pos')

    return predition_step(model, text_tensor, device)



