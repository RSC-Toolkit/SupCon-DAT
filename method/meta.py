import numpy as np
import torch



def polarization(meta_age_loss, meta_sex_loss, meta_dev_loss, meta_loc_loss, meta_data_loss, args):
    #print('meta_age_loss', meta_age_loss, type(meta_age_loss), meta_age_loss.item(), type(meta_age_loss.item()))
    meta_losses = [meta_age_loss.item(), meta_sex_loss.item(), meta_dev_loss.item(), meta_loc_loss.item(), meta_data_loss.item()]
    meta_loss_sum = torch.tensor(meta_losses, dtype=torch.float32)
    #print('meta_loss_sum', meta_loss_sum)
    if args.meta_scale:
        re_age_loss = meta_age_loss.item() / meta_loss_sum.sum()
        re_sex_loss = meta_sex_loss.item() / meta_loss_sum.sum()
        re_dev_loss = meta_dev_loss.item() / meta_loss_sum.sum()
        re_loc_loss = meta_loc_loss.item() / meta_loss_sum.sum()
        re_data_loss = meta_data_loss.item() / meta_loss_sum.sum()
    else:
        re_age_loss = meta_age_loss
        re_sex_loss = meta_sex_loss
        re_dev_loss = meta_dev_loss
        re_loc_loss = meta_loc_loss
        re_data_loss = meta_data_loss
    '''
    print('re_age_loss', re_age_loss)
    print('re_sex_loss', re_sex_loss)
    print('re_dev_loss', re_dev_loss)
    print('re_loc_loss', re_loc_loss)
    print('re_data_loss', re_data_loss)
    
    print('meta_age_loss_tmp', meta_age_loss * re_age_loss, type(meta_age_loss * re_age_loss))
    print('meta_sex_loss_tmp', meta_sex_loss * re_sex_loss, type(meta_sex_loss * re_sex_loss))
    print('meta_dev_loss_tmp', meta_dev_loss * re_dev_loss, type(meta_dev_loss * re_dev_loss))
    print('meta_loc_loss_tmp', meta_loc_loss * re_loc_loss, type(meta_loc_loss * re_loc_loss))
    print('meta_data_loss_tmp', meta_data_loss * re_data_loss, type(meta_data_loss * re_data_loss))
    '''
    output_loss = (meta_age_loss * re_age_loss) + (meta_sex_loss * re_sex_loss) + (meta_dev_loss * re_dev_loss) + (meta_loc_loss * re_loc_loss) + (meta_data_loss * re_data_loss)
    return output_loss

def equalization(meta_age_loss, meta_sex_loss, meta_dev_loss, meta_loc_loss, meta_data_loss, args):
    equal_meta_age_loss = 1 / meta_age_loss
    equal_meta_sex_loss = 1 / meta_sex_loss
    equal_meta_dev_loss = 1 / meta_dev_loss
    equal_meta_loc_loss = 1 / meta_loc_loss
    equal_meta_data_loss = 1 / meta_data_loss
    
    equal_five_loss = 1 / (equal_meta_age_loss + equal_meta_sex_loss + equal_meta_dev_loss + equal_meta_loc_loss + equal_meta_data_loss)
    #print('equal_five_loss', equal_five_loss)
    final_equal_five_loss = equal_five_loss * 5
    #print('final_equal_five_loss', final_equal_five_loss)
    
    return final_equal_five_loss

def equalization_ver2(meta_age_loss, meta_sex_loss, meta_dev_loss, meta_loc_loss, meta_data_loss, args):
    loss_magnitude_sum = meta_age_loss + meta_sex_loss + meta_dev_loss + meta_loc_loss + meta_data_loss
    meta_age_relative_ratio = meta_age_loss / loss_magnitude_sum
    meta_sex_relative_ratio = meta_sex_loss / loss_magnitude_sum
    meta_dev_relative_ratio = meta_dev_loss / loss_magnitude_sum
    meta_loc_relative_ratio = meta_loc_loss / loss_magnitude_sum
    meta_data_relative_ratio = meta_data_loss / loss_magnitude_sum
    
    alpha = 1/(1/meta_age_relative_ratio
         + 1/meta_sex_relative_ratio
         + 1/meta_dev_relative_ratio
         + 1/meta_loc_relative_ratio
         + 1/meta_data_relative_ratio)
    
    meta_age_weight = (1/meta_age_relative_ratio) * alpha
    meta_sex_weight = (1/meta_sex_relative_ratio) * alpha
    meta_dev_weight = (1/meta_dev_relative_ratio) * alpha
    meta_loc_weight = (1/meta_loc_relative_ratio) * alpha
    meta_data_weight = (1/meta_data_relative_ratio) * alpha
    
    final_loss = (meta_age_loss * meta_age_weight) + (meta_sex_loss * meta_sex_weight) + (meta_dev_loss * meta_dev_weight) + (meta_loc_loss * meta_loc_weight) + (meta_data_loss * meta_data_weight)
    return final_loss