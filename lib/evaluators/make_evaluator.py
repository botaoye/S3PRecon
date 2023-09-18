import imp


def _evaluator_factory(cfg):
    module = cfg.evaluator_module
    path = cfg.evaluator_path
    evaluator = imp.load_source(module, path).Evaluator()
    return evaluator


def make_evaluator(cfg):
    return _evaluator_factory(cfg)
