import numpy as np

from detector.object_class import ObjectClassType


class Projector:
    def __init__(self, config):
        self.config = config
        print('<<<<<<<<<<<<<<<<<<<<<<<<<PROJECTOR>>>>>>>>>>>>>>>>>>>>>>>>>')
        self._previous_projections_dict = {}

    @staticmethod
    def _frame_projections_to_points_json(projections, radius, class_type):
        points = []
        print("<<<<<<<<<<<<<<<<<<<<<FRAME_PROJECTIONs_to_points_json>>>>>>>>>>>>>>>>>>>>>>")
        for x, y, opacity in projections:
            points.append({
                "x": x,
                "y": y,
                "opacity": opacity,
                "radius": radius,
                "classType": class_type
            })

        return points

    def _decay_projections(self, cls):
        print('<<<<<<<<<<<<<<<<<<<<<<<<<_decay_projections>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        self._previous_projections_dict[cls][:, 2] -= \
            1.0 / self.config.decay_point_time_sec / self.config.fps

        self._previous_projections_dict[cls] = np.delete(
            self._previous_projections_dict[cls],
            np.argwhere(self._previous_projections_dict[cls][:, 2] <= 0).flatten(),
            axis=0
        )

    def get_next_projection_batch(self, data, out):
        result = []
        print('<<<<<<<<<<<<<<<<<<get_next_projection_batch>>>>>>>>>>>>>>>')
        frame_id_video_id_dict = []
        #for frame_id in frame_id_video_id_dict:
        #    result.append({
        #        "frameId": frame_id,
        #        "points": []
        #    })

        projections_dict = {}
        cls = 'car'
        #for video_id in frame_id_video_id_dict[frame_id]:
        #for src_name in data.get_source_names():
        idx = 0
        for i in out:
            #meta_frames = data.get_meta_frames_by_src_name(src_name)
            #for meta_frame in meta_frames:
            #print('METABOX', meta_frame.get_bbox_info())
            #for x, y, _, cls, idx in frame_id_video_id_dict[frame_id][video_id]:
            #for box in meta_frame.get_bbox_info():
            for box in i['boxes']:
                print('BOX', box)
                r = 0.0
                if cls == ObjectClassType.PERSON.value:
                    r = self.config.person_radius * 2.0
                if cls == ObjectClassType.CAR.value:
                    r = self.config.car_radius * 2.0
                x = ((box[0] + box[2]) / 2).cpu().detach().numpy().astype(np.uint32)
                y = ((box[1] + box[3]) / 2).cpu().detach().numpy().astype(np.uint32)
                video_id = 0
                if cls not in self._previous_projections_dict:
                    self._previous_projections_dict[cls] = np.array([])

                #if cls not in projections_dict:
                if True:
                    projections_dict[cls] = {
                        "projections": np.array([]).reshape((0, 5)),
                        "projection_cnt": 0,
                        "radius": r
                    }


                    projections_dict[cls]["projection_cnt"] += 1
                    print('TYPES', x, y)
                    projections_dict[cls]["projections"] = np.append(
                        projections_dict[cls]["projections"],
                        [[x, y, cls, idx, video_id]],
                        axis=0
                    )
                idx += 1

            for cls in projections_dict:
                projections = projections_dict[cls]["projections"]
                projection_cnt = projections_dict[cls]["projection_cnt"]
                radius = projections_dict[cls]["radius"]

                if projection_cnt == 0:
                    break

                video_id_array = projections[:, 4]

                point_array = projections[:, :2]
                print('PROJECTIONS', projections)
                dist_matrix = np.linalg.norm(
                    np.repeat([projections[:, :2]], projection_cnt, axis=0).reshape((projection_cnt ** 2, 2)) -
                    np.repeat(projections[:, :2], projection_cnt, axis=0),
                    axis=1
                ).reshape((projection_cnt, projection_cnt))

                same_video_id_matrix = (
                        np.repeat([video_id_array], projection_cnt, axis=0).flatten() ==
                        np.repeat(video_id_array, projection_cnt, axis=0)
                ).reshape((projection_cnt, projection_cnt))

                dist_matrix[same_video_id_matrix] = np.inf

                while True:
                    min_dist = np.min(dist_matrix)

                    if min_dist > radius:
                        break

                    min_pos = np.argmin(dist_matrix)
                    min_i = np.floor(min_pos / dist_matrix.shape[0]).astype(np.int64)
                    min_j = (min_pos % dist_matrix.shape[1]).astype(np.int64)

                    mean_point = np.mean([point_array[min_j], point_array[min_i]], axis=0)
                    point_array = np.append(point_array, [mean_point], axis=0)

                    dist_matrix = np.append(dist_matrix, [[0] * dist_matrix.shape[1]], axis=0)
                    dist_matrix = np.append(dist_matrix.T, [[0] * dist_matrix.T.shape[1]], axis=0).T

                    new_dist_array = np.linalg.norm(point_array - mean_point, axis=1)
                    new_dist_array[np.argwhere(dist_matrix[min_i, :] == np.inf).flatten()] = np.inf
                    new_dist_array[np.argwhere(dist_matrix[min_j, :] == np.inf).flatten()] = np.inf

                    dist_matrix[-1, :] = new_dist_array
                    dist_matrix[:, -1] = new_dist_array

                    dist_matrix = np.delete(dist_matrix, np.min([min_i, min_j]), axis=0)
                    dist_matrix = np.delete(dist_matrix, np.max([min_i, min_j]), axis=0)

                    dist_matrix = np.delete(dist_matrix, np.min([min_i, min_j]), axis=1)
                    dist_matrix = np.delete(dist_matrix, np.max([min_i, min_j]), axis=1)

                    point_array = np.delete(point_array, np.max([min_i, min_j]), axis=0)
                    point_array = np.delete(point_array, np.min([min_i, min_j]), axis=0)

                point_array = np.vstack([point_array.T, [[1] * point_array.shape[0]]]).T

                projections_dict[cls] = {
                    "projections": point_array,
                    "projection_cnt": point_array.shape[0],
                    "radius": radius
                }

            for cls in projections_dict:
                radius = projections_dict[cls]["radius"]
                class_type = None
                if cls == ObjectClassType.PERSON.value:
                    class_type = "PERSON"
                if cls == ObjectClassType.CAR.value:
                    class_type = "CAR"

                if cls not in self._previous_projections_dict or self._previous_projections_dict[cls].shape[0] == 0:
                    self._previous_projections_dict[cls] = projections_dict[cls]["projections"]

                    result[-1]["points"].extend(self._frame_projections_to_points_json(
                        projections_dict[cls]["projections"],
                        radius,
                        class_type
                    ))
                else:
                    previous_projections = self._previous_projections_dict[cls]
                    previous_projection_cnt = previous_projections.shape[0]

                    projections = projections_dict[cls]["projections"]
                    projection_cnt = projections_dict[cls]["projection_cnt"]

                    mean_distance_per_frame = 0.0
                    if cls == ObjectClassType.PERSON.value:
                        mean_distance_per_frame = self.config.person_mean_distance_per_frame()
                    if cls == ObjectClassType.CAR.value:
                        mean_distance_per_frame = self.config.car_mean_distance_per_frame()

                    dist_matrix = np.linalg.norm(
                        np.repeat([previous_projections[:, :-1]], projection_cnt, axis=0).reshape(
                            (previous_projection_cnt * projection_cnt, 2)) -
                        np.repeat(projections[:, :-1], previous_projection_cnt, axis=0),
                        axis=1
                    ).reshape((projection_cnt, previous_projection_cnt))

                    where_correlation_arg = np.argwhere(dist_matrix <= mean_distance_per_frame)

                    if where_correlation_arg.shape[0] == 0:
                        self._decay_projections(cls)

                        self._previous_projections_dict[cls] = np.vstack([
                            self._previous_projections_dict[cls],
                            projections
                        ])

                        result[-1]["points"].extend(self._frame_projections_to_points_json(
                            projections,
                            radius,
                            class_type
                        ))

                        continue

                    correlations = np.vstack([
                        dist_matrix[where_correlation_arg[:, 0], where_correlation_arg[:, 1]],
                        where_correlation_arg[:, 0],
                        where_correlation_arg[:, 1]
                    ])

                    correlations = correlations[:, np.argsort(correlations[0, :])]
                    correlations = correlations[1:, :]
                    correlations = correlations.astype(np.int64)

                    correlation_matrix = np.full((projection_cnt, previous_projection_cnt), False)

                    for i, j in correlations.T:
                        if np.ma.any(correlation_matrix[i, :]) or np.ma.any(correlation_matrix[:, j]):
                            continue

                        correlation_matrix[i, j] = True

                    self._previous_projections_dict[cls] = np.delete(
                        self._previous_projections_dict[cls],
                        np.any(correlation_matrix, axis=0),
                        axis=0
                    )

                    self._decay_projections(cls)

                    projections_correlations = np.stack([projections[:, :-1], projections[:, :-1]], axis=1)
                    where_correlation_arg = np.argwhere(correlation_matrix)
                    projections_correlations[where_correlation_arg[:, 0], 1, :] = \
                        previous_projections[where_correlation_arg[:, 1], :-1]

                    new_projections = np.mean(projections_correlations, axis=1)
                    new_projections = np.vstack([
                        new_projections.T,
                        [1] * new_projections.shape[0]
                    ]).T
                    self._previous_projections_dict[cls] = np.vstack([
                        self._previous_projections_dict[cls],
                        new_projections
                    ])

                    result[-1]["points"].extend(
                        self._frame_projections_to_points_json(new_projections, radius, class_type)
                    )

        return result
