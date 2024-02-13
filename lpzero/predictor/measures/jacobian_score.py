from lpzero.predictor.measures.jacobian_covariance import cosine, covariance
import torch 

def jacobian_score(model, inputs):
    output = model(**inputs).last_hidden_state
    output.backward(torch.ones_like(output))
    jacobs = model.embeddings.position_embeddings.weight.grad.detach()
    return covariance(jacobs)


def jacobian_score_cosine(model, inputs):
    output = model(**inputs).last_hidden_state
    output.backward(torch.ones_like(output))
    jacobs = model.embeddings.position_embeddings.weight.grad.detach()
    return cosine(jacobs)
