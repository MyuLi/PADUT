def set_template(args):
    if args.tune:
        args.max_epoch = 20
        args.milestones = [7, 12, 15, 18]
        args.scheduler = 'MultiStepLR'

    if args.template.find('duf_mixs2') >= 0: 
        args.input_setting = 'Y'
        args.input_mask = 'Phi'
        if args.stage == 9: args.batch_size = 1
        elif args.stage == 7: args.batch_size = 1
        args.learning_rate = min(2e-4 * args.batch_size, 4e-4)

    if args.template.find('dauhst') >= 0:
        args.input_setting = 'Y'
        args.input_mask = 'Phi_PhiPhiT'
        #args.learning_rate = min(2e-4 * args.batch_size, 4e-4)

        
