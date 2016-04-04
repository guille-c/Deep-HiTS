from layers import *

class ArchBuilder():
    def __init__(self, arch, input_data, params, rng):
        self.params = params
        self.i_params = 0
        self.rng = rng
        self.layers = []
        dataInfo = arch[0]
        print dataInfo
        assert dataInfo["layer"]=="DataLayer"
        arch = arch[1:]# push out DataLayer
        input_shape = [dataInfo["batch_size"],
                      dataInfo["im_chan"],
                      dataInfo["im_size"],
                      dataInfo["im_size"]]
        self.layers.append(DataLayer(input_data, input_shape))
        for layerInfo in arch:
            self.addLayer(layerInfo)

    def addLayer(self, layerInfo):
        layerType = layerInfo["layer"]
        input_shape = self.layers[-1].getOutputShape()
        print "input_shape: ", input_shape
        print layerInfo
        if layerType=="CreateRotationsLayer":
            self.layers.append(CreateRotationsLayer(self.layers[-1].output,
                                                    input_shape))
        elif layerType=="ConvPoolLayer":
            filter_shape = [layerInfo["num_output"],
                            input_shape[1],
                            layerInfo["filter_size"],
                            layerInfo["filter_size"]
            ]
            W = self.params[self.i_params]
            self.i_params += 1
            b = self.params[self.i_params]
            self.i_params += 1
            self.layers.append(LeNetConvPoolLayer(self.rng,
                                                  input=self.layers[-1].output,
                                                  filter_shape=filter_shape,
                                                  image_shape=input_shape,
                                                  pad=layerInfo["pad"],
                                                  poolsize=(layerInfo["pool_size"],
                                                            layerInfo["pool_size"]),
                                                  activation = layerInfo["activation"],
                                                  poolstride=(layerInfo["pool_size"],
                                                            layerInfo["pool_size"]),
                                                  init_type=layerInfo["init_type"],
                                                  W=W,
                                                  b=b))
        elif layerType=="JoinRotationsLayer":
            self.layers.append(JoinRotationsLayer(self.layers[-1].output,
                                                  input_shape))
        elif layerType=="HiddenLayer":
            W = self.params[self.i_params]
            self.i_params += 1
            b = self.params[self.i_params]
            self.i_params += 1
            self.layers.append(HiddenLayer(
                self.rng,
                input=self.layers[-1].output,
                n_in=input_shape[1],
                n_out=layerInfo["num_output"],
                activation=layerInfo["activation"],
                init_type=layerInfo["init_type"],
                W=W,
                b=b))
            self.layers[-1].setBatchSize(input_shape[0])
        elif layerType=="DropoutLayer":
            self.layers.append(DropoutLayer(self.layers[-1].output,
                                            p_drop = layerInfo["p_drop"]))
            self.layers[-1].setOutputShape(input_shape)
        elif layerType=="LogisticRegression":
            W = self.params[self.i_params]
            self.i_params += 1
            b = self.params[self.i_params]
            self.i_params += 1
            self.layers.append(LogisticRegression(input=self.layers[-1].output,
                                                  n_in=input_shape[1],
                                                  n_out=layerInfo["num_output"],
                                                  init_type=layerInfo["init_type"],
                                                  W=W,
                                                  b=b,
                                                  rng=self.rng))
            self.layers[-1].setBatchSize(input_shape[0])
        elif layerType=="ResidualLayer":
            filter_shape = [layerInfo["num_output"],
                            input_shape[1],
                            layerInfo["filter_size"],
                            layerInfo["filter_size"]
            ]
            W1 = self.params[self.i_params]
            self.i_params += 1
            b1 = self.params[self.i_params]
            self.i_params += 1
            W2 = self.params[self.i_params]
            self.i_params += 1
            b2 = self.params[self.i_params]
            self.i_params += 1
            self.layers.append(ResidualLayer(rng=self.rng,
                                             input=self.layers[-1].output,
                                             filter_shape=filter_shape,
                                             image_shape=input_shape,
                                             activation=layerInfo["activation"],
                                             W1 = W1,
                                             b1 = b1,
                                             W2 = W2,
                                             b2 = b2))                                        
        elif layerType=="MaxPoolingLayer":
            self.layers.append(MaxPoolingLayer(input=self.layers[-1].output,
                                               image_shape = input_shape,
                                               size = layerInfo["size"],
                                               stride = layerInfo["stride"]))                           
        else:
            raise Exception("layerType "+str(layerType)+" is not valid")
        print "Output shape ",self.layers[-1].getOutputShape()
        print "-----------------"
    def getLayers(self):
        return self.layers
    def getParams(self):
        params = []
        for layer in self.layers:
            if layer.hasParams():
                params = params+layer.params
        return params
    def loadParams(self, params):
        i = 0
        for layer in self.layers:
            nParams = layer.nParams()
            if nParams==2:
                layer.load_params(params[i],params[i+1])
                i+=2
            elif nParams==4:
                layer.load_params(params[i],
                                  params[i+1],
                                  params[i+2],
                                  params[i+3])
                i+=4
            elif nParams==0:
                pass
            else:
                raise Exception("nParams==",nParams)
