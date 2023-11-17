import os

ignore_list = ['7WTilOM9F-0_21_0', 'C83WOaDDolU_22_0', 'C83WOaDDolU_23_0', '4FhT4GcBgpE_44_3', '0nGvFc8QHMo_89_5', '0nGvFc8QHMo_90_0', '0nGvFc8QHMo_90_5', '4nd7Qj9MEHU_14_0', 'guKnv7AjmCg_6_1', 'exXxZa6x2HA_45_2', 'DbT5_QuCBis_18_0', 'DL5ebsl88iQ_22_9', 'kcE7zoSu9HY_1_0', 'DD_E3taRvBg_12_0', 'iwBqTovcWJM_5_0', 'jcDz_5CrJTQ_0_0', 'jcDz_5CrJTQ_4_0', 'hbyffqXb8wY_3_0', 'hZ2QJXRX_GI_63_0', 'g_ZGcK4uWMA_4_0', 'LA-f6sle_yw_11_0', 'LA-f6sle_yw_9_0', 'SfFAohd5q-Q_17_0', 'SfFAohd5q-Q_19_0', 'rRRSWmZrJGs_24_1', 'ocNxd2xDr38_24_0', 'mTNlVAz2fdA_23_4', 'mTNlVAz2fdA_24_1', 'mTNlVAz2fdA_24_2', 'p27200f_mxQ_4_0', 'Piruqx4YxVw_37_0', 'rlWuO_gPLA0_8_0', 'rlWuO_gPLA0_9_0', 'uqLghyB07I4_16_0', 'vh_Xh0Z6ujY_8_1', 'Whseu8CPHmY_21_1', 'Whseu8CPHmY_22_0', '_1qKQfAyGqI_15_0', 'YLvzXQF6B90_45_0', 'YLvzXQF6B90_46_0']

# path = 'C:\\Users\\jisoo.kim\\Desktop\\Meshtalk\\audio_crop\\'
# path = 'C:\Users\jisoo.kim\Desktop\Meshtalk\audio_crop\'

for ignore in ignore_list :
    # os.remove(f'{path}{ignore}.wav')
    path = f'C:\\Users\\jisoo.kim\\Desktop\\Meshtalk\\audio_crop\\{ignore}.wav'
    os.remove(path)
    # os.remove(path+ignore+'.mp4')
    print(f'removed {path}{ignore}.wav')