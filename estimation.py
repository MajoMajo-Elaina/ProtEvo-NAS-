from DPFunc.stackgcn import StackGcn
class Estimation(object):
    
    def __init__(self,
                 gnn_architecture,
                 gnn_parameter):

        if not isinstance(gnn_architecture, list):
            raise Exception("gnn_architecture Class Wrong, require list Class ", "but input Class:",
                            type(gnn_architecture))

        if not isinstance(gnn_parameter, dict):
            raise Exception("gnn_parameter Class Wrong, require dict Class ", "but input Class:",
                            type(gnn_parameter))

        self.gnn_architecture = gnn_architecture
        self.gnn_parameter = gnn_parameter

        self.gnn_parameter = gnn_parameter

    def get_performance(self):
        data_cnf=self.gnn_parameter['data_cnf']
        gpu_number=self.gnn_parameter['gpu_number']
        epoch_number=self.gnn_parameter['epoch_number']
        pre_name=self.gnn_parameter['pre_name']
        model = StackGcn(self.gnn_architecture,
                 data_cnf,
                 gpu_number,
                 epoch_number,
                 pre_name
                         )
        performance = model.fit()
        return performance