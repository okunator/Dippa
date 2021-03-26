import src.dl.inference.post_processing as post_proc


class PostProcBuilder:
    """
    Class used to initialize the the post-processor. 
    """
    @classmethod
    def set_postprocessor(cls, 
                          post_proc_method: str,
                          thresh_method: str="naive",
                          thresh: float=0.5):
        """
        Init the post-processor

        Args:
        ----------
            post_proc_method (str):
                The post processing pipeline to use.
                One of ("hover", "dist", "contour", "cellpose", "basic")
            thresh_method (str, default="naive"):
                Thresholding method for the soft masks from the instance branch.
                One of ("naive", "argmax", "sauvola", "niblack")).
            thresh (float, default = 0.5): 
                threshold probability value. Only used if method == "naive"  
        """
        assert post_proc_method in ("drfns", "dcan", "dran", "cellpose", "hover", "basic")
        c = cls()
        key = post_proc.POST_PROC_LOOKUP[post_proc_method]
        return post_proc.__dict__[key](thresh_method, thresh)
