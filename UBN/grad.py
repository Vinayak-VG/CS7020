from model_wbn import resnet110

model = resnet110().to("cuda")

for name, param in model.named_parameters():
    print(name, param.weight.grad)
    quit()