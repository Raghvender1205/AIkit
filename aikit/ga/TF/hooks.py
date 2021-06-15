import tensorflow.keras.backend as K
from aikit.utils import Hook

class GetGradients(Hook):
    def __init__(self, optimizer, gradients):
        super(GetGradients, self).__init__(optimizer, 'GetGradients')
        self.gradients = gradients
    
    def __hook__(self, loss, params):
        return self.gradients
    
class UpdateAdd(Hook):
    def __init__(self, condition, name_scope):
        super(UpdateAdd, self).__init__(K, 'UpdateAdd')
        self.condition = condition
        self.name_scope = name_scope

    def __hook__(self, x, increment):
        with K.name_scope(self.name_scope):
            if not K.is_keras_tensor(increment):
                increment = K.constant(increment, dtype=K.dtype(x))
            
            increment = K.switch(self.condition, increment, K.constant(0, dtype=K.dtype(x)))
        
        return self.__original__(x, increment)
    
class UpdateSub(Hook):
    def __init__(self, condition, name_scope):
        super(UpdateSub, self).__init__(K, 'UpdateSub')
        self.condition = condition
        self.name_scope = name_scope
    
    def __hook__(self, x, decrement):
        with K.name_scope(self.name_scope):
            if not K.is_keras_tensor(decrement):
                decrement = K.constant(decrement, dtype=K.dtype(x))
            decrement = K.switch(self.condition, decrement, K.constant(0, dtype=K.dtype(x)))
        
        return self.__original__(x, decrement)
    
class Update(Hook):
    def __init__(self, condition, name_scope):
        super(Update, self).__init__(K, 'Update')
        self.condition = condition
        self.name_scope = name_scope

    def __hook__(self, x, new_x):
        with K.name_scope(self.name_scope):
            if not K.is_keras_tensor(new_x):
                new_x = K.constant(new_x, dtype=K.dtype(x))

            new_x = K.switch(self.condition, new_x, x)

        return self.__original__(x, new_x)
