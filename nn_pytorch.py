import torch

print("hello world")



def softmax(logits):
  counts = [logit.exp() for logit in logits]
  denominator = sum(counts)
  out = [c / denominator for c in counts]
  return out


value1 = torch.Tensor([0.0]).double()  ; value1.requires_grad = True
value2 = torch.Tensor([3.0]).double()  ; value2.requires_grad = True
value3 = torch.Tensor([-2.0]).double()  ; value3.requires_grad = True
value4 = torch.Tensor([1.0]).double()  ; value4.requires_grad = True


# this is the negative log likelihood loss function, pervasive in classification
logits = [value1, value2, value3, value4]
probs = softmax(logits)
print(probs)

loss = -(probs[3].log()) # dim 3 acts as the label for this input example


loss.backward()


print(logits[3].grad, "logits grad")


ans = [0.041772570515350445, 0.8390245074625319, 0.005653302662216329, -0.8864503806400986]
for dim in range(4):
  ok = 'OK' if abs(logits[dim].grad - ans[dim]) < 1e-5 else 'WRONG!'
  print(f"{ok} for dim {dim}: expected {ans[dim]}, yours returns {logits[dim].grad.data.item()}")