import src.dl.inference.post_processing as post_proc


class PostProcBuilder:
    def __init__(self, 
                 aux_branch: bool, 
                 aux_type: str=None,
                 thresh_method: str="naive",
                 thresh: float=0.5) -> None:
        """
        Set post-processor object for post-processing network output

        aux_branch (bool):
            If True, specifies that the network does have an auxiliary branch
        aux_type (str, default=None);
            One of ("hover", "dist", "contour") 
        thresh_method (str, default="naive"):
            Thresholding method for the soft masks from the instance branch.
            One of ("naive", "argmax", "sauvola", "niblack")).
        thresh (float, default = 0.5): 
            threshold probability value. Only used if method == "naive"   
        """
        assert aux_type in ("hover", "dist", "contour")
        self.aux_branch = aux_branch
        self.aux_type = aux_type

    @classmethod
    def set_postprocessor(cls, 
                          aux_branch: bool, 
                          aux_type: str=None, 
                          thresh_method: str="naive",
                          thresh: float=0.5):
        """
        Init the post-processor

        Args:
            aux_branch (bool):
                If True, specifies that the network does have an auxiliary branch
            aux_type (str, default=None);
                One of ("hover", "dist", "contour") 
            thresh_method (str, default="naive"):
                Thresholding method for the soft masks from the instance branch.
                One of ("naive", "argmax", "sauvola", "niblack")).
            thresh (float, default = 0.5): 
                threshold probability value. Only used if method == "naive"  
        """
        c = cls(aux_branch, aux_type, thresh_method, thresh)
        key = post_proc.POST_PROC_LOOKUP[c.aux_type]
        return post_proc.__dict__[key](thresh_method, thresh)
