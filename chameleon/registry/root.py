from .registry import Registry

BLOCKS = Registry('blocks')

COMPONENTS = Registry('components')

LAYERS = Registry('layers')

OPS = Registry('ops')

OPTIMIZERS = Registry('optim')

METRICS = Registry('metrics')

BACKBONES = Registry('backbones')

NECKS = Registry('necks')

FUNCTIONS = Registry('functions')


def build_block(name, **options):
    return BLOCKS.build({'name': name, **options})


def build_component(name, **options):
    return COMPONENTS.build({'name': name, **options})


def build_layer(name, **options):
    return LAYERS.build({'name': name, **options})


def build_ops(name, **options):
    return OPS.build({'name': name, **options})


def build_optimizer(name, **options):
    return OPTIMIZERS.build({'name': name, **options})


def build_metric(name, **options):
    return METRICS.build({'name': name, **options})


def build_backbone(name, **options):
    return BACKBONES.build({'name': name, **options})


def build_neck(name, **options):
    return NECKS.build({'name': name, **options})
