from lpzero.predictor.measures.jacobian_covariance import cosine, covariance
import torch 
from . import measure


from lpzero.utils.modeling_electra import ElectraLayer, ElectraModel
from lpzero.model.hf_gpt2.model_hf_gpt2 import HfGPT2, HfGPT2Flex

def jacobian_score(model, inputs):
    output = model(**inputs).last_hidden_state
    output.backward(torch.ones_like(output))
    jacobs = model.embeddings.position_embeddings.weight.grad.detach()
    return covariance(jacobs)


@measure('jacobian_cosine')
def jacobian_score_cosine(model, inputs, targets, *args, **kwargs):
    if isinstance(model, ElectraModel):
        outputs = model(**inputs).last_hidden_state
        outputs.backward(torch.ones_like(outputs))
        jacobs = model.embeddings.position_embeddings.weight.grad.detach()
    elif isinstance(model, (HfGPT2, HfGPT2Flex)):
        loss, _, _, _ = model.forward(inputs, targets, mems=None)
        loss = loss.float().mean().type_as(loss)
        loss.backward()
        jacobs = model.model.transformer.wpe.weight.grad.detach()
    return cosine(jacobs)
