# this is from main.py


elif args.method == 'am':
                features = model(args.transforms(images), args=args, alpha=alpha, training=True)
                
                if args.domain_adaptation:
                    #features = (features, domain_features) # domain_features -> ReverseLayerF
                    output = args.arc(features[0], class_labels)
                    class_loss = criterion[0](output, class_labels)
                    
                    #class_loss = criterion[0](features[0], class_labels, classifier[0])
                    
                    '''
                    output = classifier[0](features[0])
                    class_loss = criterion[0](output, class_labels)
                    '''
                    
                    meta_output = classifier[1](features[1])
                    meta_loss = criterion[1](meta_output, meta_labels)
                    
                    loss = class_loss + (args.alpha * meta_loss)
                
                elif args.domain_adaptation2:
                    output = args.arc(features[0], class_labels)
                    class_loss = criterion[0](output, class_labels)
                    
                    '''
                    output, class_loss = criterion[0](features[0], class_labels, classifier[0])
                    
                    output = classifier[0](features[0])
                    class_loss = criterion[0](output, class_labels)
                    '''
                    
                    features2 = model(args.transforms(images), args=args, alpha=alpha, training=True)
                    
                    feat1 = features[0]
                    feat2 = features2[0]
                    
                    if args.target_type == 'project_flow_all':
                        proj1, proj2 = classifier[1](feat1), classifier[1](feat2) # classifier[1] = projector #
                    
                    
                    elif args.target_type == 'representation_all':
                        proj1, proj2 = feat1, feat2
                    
                    elif args.target_type == 'z1block_project':
                        proj1 = deepcopy(feat1.detach())
                        proj2 = classifier[1](feat2)
                    
                    elif args.target_type == 'z1_project2':
                        proj1 = feat1
                        proj2 = classifier[1](feat2)
                    
                    elif args.target_type == 'project1block_project2':
                        proj1 = deepcopy(classifier[1](feat1).detach())
                        proj2 = classifier[1](feat2)
                    
                    elif args.target_type == 'project1_r2block':
                        proj1 = classifier[1](feat1)
                        proj2 = deepcopy(feat2.detach())
                    
                    elif args.target_type == 'project1_r2':
                        proj1 = classifier[1](feat1)
                        proj2 = feat2
                    
                    elif args.target_type == 'project1_project2block':
                        proj1 = classifier[1](feat1)
                        proj2 = deepcopy(classifier[1](feat2).detach())
                    
                    elif args.target_type == 'project_block_all':
                        proj1 = deepcopy(classifier[1](feat1).detach())
                        proj2 = deepcopy(classifier[1](feat2).detach())
                    
                    meta_loss = criterion[1](proj1, proj2, meta_labels) #meta cl loss
                    
                    loss = class_loss + args.alpha * meta_loss
                
                else:
                    #loss = criterion[0](features, labels)
                    #output = features
                    
                    
                    ##arcface
                    output = args.arc(features, labels)
                    loss = criterion[0](output, labels)
                    
                    #arc_ver2
                    #output, loss = criterion[0](features, labels, classifier)
                    '''
                    #previous
                    output = classifier(features)
                    loss = criterion[0](output, labels)
                    '''