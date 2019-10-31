class DataManager:
    """Base object for managing data source"""
    def __init__(self):
        pass

    def __iter__(self, *args, **kwargs):
        """When called, manager will give one instance of data pair"""
        pass

    def mini_batch(self, batch_size):
        """Return a mini-batch of data"""
        pass

    def _download(self, *args):
        """Download data from remote source"""
        pass

    def _read_data(self, **source):
        """Read data in to memory"""
        pass


