from .padut import PADUT
def model_generator(opt, device="cuda"):
    method = opt.method 
   
    if 'padut' in method:
        num_iterations = int(method.split('_')[-1])
        model = PADUT(in_c=28, n_feat=28,nums_stages=num_iterations-1).to(device)
    else:
        print(f'opt.Method {opt.method} is not defined !!!!')
    
    return model