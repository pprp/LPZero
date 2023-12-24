from lpzero.predictor.measures.jacobian_covariance import covariance, cosine

def jacobian_score(model):
    jacobs = model.embeddings.position_embeddings.weight.grad.detach()
    return covariance(jacobs)

def jacobian_score_cosine(model):
    jacobs = model.embeddings.position_embeddings.weight.grad.detach()
    return cosine(jacobs)