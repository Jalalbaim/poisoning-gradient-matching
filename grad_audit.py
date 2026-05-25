import torch
grads   = torch.load('gradient_logs/gradients_validate_epoch0022.pt')
svd     = torch.load('gradient_logs/SVD_validate_epoch0022.pt')
G_first = grads[0]['G'] 
S_first = svd[0]['singular_values'] 

print(G_first)

print(G_first.shape)

print(S_first)

print(S_first.shape)

print(grads[0]['phase'], grads[0]['is_poisoned'], grads[0]['epoch'], grads[0]['batch'])