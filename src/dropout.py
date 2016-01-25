# Code from https://github.com/uoguelph-mlrg/theano_alexnet
class DropoutLayer(object):
    seed_common = np.random.RandomState(0)  # for deterministic results
    # seed_common = np.random.RandomState()
    layers = []
    
    def __init__(self, input, n_in, n_out, prob_drop=0.5):
        
        self.prob_drop = prob_drop
        self.prob_keep = 1.0 - prob_drop
        self.flag_on = theano.shared(np.cast[theano.config.floatX](1.0))
        self.flag_off = 1.0 - self.flag_on
        
        seed_this = DropoutLayer.seed_common.randint(0, 2**31-1)
        mask_rng = theano.tensor.shared_randomstreams.RandomStreams(seed_this)
        self.mask = mask_rng.binomial(n=1, p=self.prob_keep, size=input.shape)
        
        self.output = \
            self.flag_on * T.cast(self.mask, theano.config.floatX) * input + \
            self.flag_off * self.prob_keep * input
        
        DropoutLayer.layers.append(self)
        
        print 'dropout layer with P_drop: ' + str(self.prob_drop)
        
    @staticmethod
    def SetDropoutOn():
        for i in range(0, len(DropoutLayer.layers)):
            DropoutLayer.layers[i].flag_on.set_value(1.0)
            
    @staticmethod
    def SetDropoutOff():
        for i in range(0, len(DropoutLayer.layers)):
            DropoutLayer.layers[i].flag_on.set_value(0.0)
