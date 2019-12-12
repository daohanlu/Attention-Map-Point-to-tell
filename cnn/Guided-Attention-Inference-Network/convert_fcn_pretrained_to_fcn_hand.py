import os
import fcn
import chainer

from models.fcn8_hand_v2 import FCN8s_hand

if __name__ == '__main__':
    model_own = FCN8s_hand()
    model_original = fcn.models.FCN8s()
    model_file = fcn.models.FCN8s.download()
    chainer.serializers.load_npz(model_file, model_original)

    ignored_layers = {'score_fr', 'upscore2', 'upscore8', 'score_pool3', 'score_pool4', 'upscore_pool4'}
    print("Copying layers from pretrained fcn8s to fcn8s_hand")
    for layers in model_original._children:
        if layers not in ignored_layers:
            print('Copying {}'.format(layers))
            assert str(getattr(model_original, layers)) == str(getattr(model_own, layers)), 'Layer shape mismatch for layer {}!\noriginal: {}\nNew:{}'\
                .format(layers, getattr(model_original, layers), getattr(model_own, layers))
            setattr(model_own, layers, getattr(model_original, layers))
        else:
            print("Ignored copying attributes from layer {}".format(layers))
    print('\nSaving...')
    chainer.serializers.save_npz('fcn8s_hand_gain_pretrained.npz', model_own)