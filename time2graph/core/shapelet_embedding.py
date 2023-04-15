# -*- coding: utf-8 -*-
from .shapelet_utils import *
embed_number = 5


def time_series_embeds_factory__(embed_size, embeddings, threshold,
                                 multi_graph, debug, mode):
    def __concate__(pid, args, queue):
        ret = []
        for sdist in args:
            tmp = np.zeros(len(sdist) * embed_size * embed_number, dtype=np.float32).reshape(-1)
            for sidx in range(len(sdist)):
                dist = sdist[sidx, :]
                target = np.argsort(np.argwhere(dist <= threshold).reshape(-1))[:embed_number]
                if len(target) == 0:
                    continue
                weight = 1.0 - minmax_scale(dist[target])
                if np.sum(weight) == 0:
                    Debugger.warn_print(msg='dist {}, weight {}'.format(dist, weight), debug=debug)
                else:
                    weight /= np.sum(weight)
                target_number = len(weight)
                for k in range(target_number):
                    src, dst = (sidx * embed_number + k) * embed_size, (sidx * embed_number + k + 1) * embed_size
                    if multi_graph:
                        if sidx == 0:
                            tmp[src: dst] = weight[k] * embeddings[sidx, target[k]].reshape(-1)
                        elif sidx == len(sdist) - 1:
                            tmp[src: dst] = weight[k] * embeddings[sidx - 1, target[k]].reshape(-1)
                        else:
                            former = weight[k] * embeddings[sidx - 1, target[k]].reshape(-1)
                            latter = weight[k] * embeddings[sidx, target[k]].reshape(-1)
                            tmp[src: dst] = (former + latter)
                    else:
                        tmp[src: dst] = weight[k] * embeddings[0, target[k]].reshape(-1)
            ret.append(tmp)
            queue.put(0)
        return ret

    def __aggregate__(pid, args, queue):
        ret = []
        for sdist in args:
            tmp = np.zeros(len(sdist) * embed_size, dtype=np.float32).reshape(-1)
            for sidx in range(len(sdist)):
                dist = sdist[sidx, :]
                target = np.argsort(np.argwhere(dist <= threshold).reshape(-1))[:embed_number]
                if len(target) == 0:
                    continue
                weight = 1.0 - minmax_scale(dist[target])
                if np.sum(weight) == 0:
                    Debugger.warn_print(msg='dist {}, weight {}'.format(dist, weight), debug=debug)
                else:
                    weight /= np.sum(weight)
                src, dst = sidx * embed_size, (sidx + 1) * embed_size
                for k in range(len(weight)):
                    if multi_graph:
                        if sidx == 0:
                            tmp[src: dst] += weight[k] * embeddings[sidx, target[k]].reshape(-1)
                        elif sidx == len(sdist) - 1:
                            tmp[src: dst] += weight[k] * embeddings[sidx - 1, target[k]].reshape(-1)
                        else:
                            former = weight[k] * embeddings[sidx - 1, target[k]].reshape(-1)
                            latter = weight[k] * embeddings[sidx, target[k]].reshape(-1)
                            tmp[src: dst] += (former + latter)
                    else:
                        tmp[src: dst] += weight[k] * embeddings[0, target[k]].reshape(-1)
            ret.append(tmp)
            queue.put(0)
        return ret

    if mode == 'concate':
        return __concate__
    elif mode == 'aggregate':
        return __aggregate__
    else:
        raise NotImplementedError('unsupported mode {}'.format(mode))


class ShapeletEmbedding(object):
    def __init__(self, seg_length, tflag, multi_graph, cache_dir,
                 percentile, tanh, debug, measurement, mode,
                 **deepwalk_args):
        self.seg_length = seg_length
        self.tflag = tflag
        self.multi_graph = multi_graph
        self.cache_dir = cache_dir
        self.tanh = tanh
        self.debug = debug
        self.percentile = percentile
        self.dist_threshold = -1
        self.measurement = measurement
        self.mode = mode
        self.deepwalk_args = deepwalk_args
        self.embed_size = self.deepwalk_args.get('representation_size', 256)
        self.embeddings = None

    def fit(self, time_series_set, shapelets, warp, init=0):
        Debugger.info_print('fit shape: {}'.format(time_series_set.shape))
        tmat, sdist, dist_threshold = transition_matrix(
            time_series_set=time_series_set, shapelets=shapelets, seg_length=self.seg_length,
            tflag=self.tflag, multi_graph=self.multi_graph, tanh=self.tanh, debug=self.debug,
            init=init, warp=warp, percentile=self.percentile, threshold=self.dist_threshold,
            measurement=self.measurement)
        self.dist_threshold = dist_threshold
        self.embeddings = graph_embedding(
            tmat=tmat, num_shapelet=len(shapelets), embed_size=self.embed_size,
            cache_dir=self.cache_dir, **self.deepwalk_args)

    def time_series_embedding(self, time_series_set, shapelets, warp, init=0):
        if self.embeddings is None:
            self.fit(time_series_set=time_series_set, shapelets=shapelets, warp=warp)
        sdist = shapelet_distance(time_series_set=time_series_set, shapelets=shapelets,
                                  seg_length=self.seg_length, tflag=self.tflag, tanh=self.tanh,
                                  debug=self.debug, init=init, warp=warp,
                                  measurement=self.measurement)
        Debugger.info_print('embedding threshold {}'.format(self.dist_threshold))
        Debugger.info_print('sdist size {}'.format(sdist.shape))
        parmap = ParMap(
            work=time_series_embeds_factory__(
                embed_size=self.embed_size, embeddings=self.embeddings, threshold=self.dist_threshold,
                multi_graph=self.multi_graph, debug=self.debug, mode=self.mode),
            monitor=parallel_monitor(msg='time series embedding', size=sdist.shape[0], debug=self.debug),
            njobs=NJOBS
        )
        if self.mode == 'concate':
            size = sdist.shape[1] * self.embed_size * embed_number
        elif self.mode == 'aggregate':
            size = sdist.shape[1] * self.embed_size
        else:
            raise NotImplementedError('unsupported mode {}'.format(self.mode))
        return np.array(parmap.run(data=list(sdist)), dtype=np.float32).reshape(sdist.shape[0], size)
