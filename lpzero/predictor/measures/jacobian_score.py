from lpzero.predictor.measures.jacobian_covariance import cosine, covariance
import torch 
from . import measure

def jacobian_score(model, inputs):
    output = model(**inputs).last_hidden_state
    output.backward(torch.ones_like(output))
    jacobs = model.embeddings.position_embeddings.weight.grad.detach()
    return covariance(jacobs)


@measure('jacobian_cosine')
def jacobian_score_cosine(model, inputs, *args, **kwargs):
    output = model(**inputs).last_hidden_state
    output.backward(torch.ones_like(output))
    jacobs = model.embeddings.position_embeddings.weight.grad.detach()
    return cosine(jacobs)
