import os

from lib.config import args, cfg
from lib.datasets.make_dataset import make_render_data_loader
from lib.evaluators.eval_render import eval_render, render_time_test
from lib.utils.mesh_utils import refuse_ba, refuse_gt


def run_gt_mesh_extract():
    from lib.datasets import make_data_loader
    from lib.utils.mesh_utils import extract_mesh, refuse, transform
    import open3d as o3d

    data_loader = make_data_loader(cfg, is_train=False)

    mesh = refuse_gt(data_loader)
    # mesh = transform(mesh, cfg.test_dataset.scale, cfg.test_dataset.offset)

    assert args.output_mesh != ''
    o3d.io.write_triangle_mesh(args.output_mesh, mesh)


def run_render_mesh():
    from lib.datasets import make_data_loader
    import open3d as o3d
    from tqdm import tqdm
    import numpy as np
    import imageio

    data_loader = make_data_loader(cfg, is_train=False)
    intrinsics = data_loader.dataset.intrinsic

    H, W = 480, 640
    print("=> Load mesh: {}".format(args.render_mesh))
    geometry = o3d.io.read_triangle_mesh(args.render_mesh)
    geometry.compute_vertex_normals()
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=W, height=H, visible=False)
    ctrl = vis.get_view_control()
    vis.add_geometry(geometry)
    # opt = vis.get_render_option()
    # opt.mesh_show_back_face = True

    cam = ctrl.convert_to_pinhole_camera_parameters()
    intr = intrinsics.data.cpu().numpy()
    # cam.intrinsic.set_intrinsics(W, H, intr[0,0], intr[1,1], intr[0,2], intr[1,2])
    cam.intrinsic.set_intrinsics(W, H, intr[0, 0], intr[1, 1], W / 2 - 0.5, H / 2 - 0.5)
    ctrl.convert_from_pinhole_camera_parameters(cam)

    mesh_imgs = []
    for iteration, batch in enumerate(tqdm(data_loader, desc='Rendering')):
        # file_name = batch['meta']['filename'][0]
        c2w = batch['c2w']

        extr = np.linalg.inv(c2w)
        cam.extrinsic = extr
        ctrl.convert_from_pinhole_camera_parameters(cam)
        vis.poll_events()
        vis.update_renderer()
        # if not args.debug:
        rgb_mesh = vis.capture_screen_float_buffer(do_render=True)
        mesh_imgs.append(np.asarray(rgb_mesh))

    def integerify(img):
        return (img*255.).astype(np.uint8)
    mesh_imgs = [integerify(img) for img in mesh_imgs]

    vis.destroy_window()
    imageio.mimwrite('mesh.mp4', mesh_imgs, fps=args.fps, quality=10)
    # imageio.mimwrite(os.path.join('out', '{}_mesh_{}.mp4'.format(outbase, post_fix)), mesh_imgs, fps=args.fps,
    #                  quality=10)

    # mesh = extract_mesh(network.model.sdf_net)
    # mesh = refuse(mesh, data_loader)
    # mesh = transform(mesh, cfg.test_dataset.scale, cfg.test_dataset.offset)
    #
    # assert args.output_mesh != ''
    # o3d.io.write_triangle_mesh(args.output_mesh, mesh)


def run_mesh_extract():
    from lib.datasets import make_data_loader
    from lib.networks import make_network
    from lib.utils.mesh_utils import extract_mesh, refuse, transform
    from lib.utils.net_utils import load_network
    import open3d as o3d

    network = make_network(cfg).cuda()
    load_network(
        network,
        cfg.trained_model_dir,
        resume=cfg.resume,
        epoch=cfg.test.epoch
    )
    network.eval()
    data_loader = make_data_loader(cfg, is_train=False)

    mesh = extract_mesh(network.model.sdf_net)
    mesh = refuse(mesh, data_loader)
    mesh = transform(mesh, cfg.test_dataset.scale, cfg.test_dataset.offset)

    assert args.output_mesh != ''
    o3d.io.write_triangle_mesh(args.output_mesh, mesh)


def print_result(result_dict):
    for k, v in result_dict.items():
        print(f'{k:7s}: {v:1.3f}')


def run_evaluate():
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    from lib.networks import make_network
    from lib.utils.mesh_utils import extract_mesh, refuse, transform
    from lib.utils.net_utils import load_network
    import open3d as o3d

    network = make_network(cfg).cuda()
    load_network(
        network,
        cfg.trained_model_dir,
        resume=cfg.resume,
        epoch=cfg.test.epoch
    )
    network.eval()
    data_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)

    mesh = extract_mesh(network.model.sdf_net)
    if 'ba' in cfg.network_module:
        # refuse with camera pose refinement
        mesh = refuse_ba(mesh, data_loader, network)
    else:
        mesh = refuse(mesh, data_loader)
    mesh = transform(mesh, cfg.test_dataset.scale, cfg.test_dataset.offset)

    if args.output_mesh != '':
        o3d.io.write_triangle_mesh(args.output_mesh, mesh)

    mesh_gt = o3d.io.read_triangle_mesh(f'{cfg.test_dataset.data_root}/{cfg.test_dataset.scene}/gt.obj')
    evaluate_result = evaluator.evaluate(mesh, mesh_gt)
    print_result(evaluate_result)


def run_evaluate_all():
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    from lib.networks import make_network
    from lib.utils.mesh_utils import extract_mesh, refuse, transform
    from lib.utils.net_utils import load_network
    import open3d as o3d

    data_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)

    for epoch in range(9, 50):
        print('eval epoch: ', epoch)
        network = make_network(cfg).cuda()
        load_network(
            network,
            cfg.trained_model_dir,
            resume=cfg.resume,
            epoch=epoch
        )
        network.eval()

        mesh = extract_mesh(network.model.sdf_net)
        if 'ba' in cfg.network_module:
            # refuse with camera pose refinement
            mesh = refuse_ba(mesh, data_loader, network)
        else:
            mesh = refuse(mesh, data_loader)
        mesh = transform(mesh, cfg.test_dataset.scale, cfg.test_dataset.offset)

        if args.output_mesh != '':
            o3d.io.write_triangle_mesh(args.output_mesh, mesh)

        mesh_gt = o3d.io.read_triangle_mesh(f'{cfg.test_dataset.data_root}/{cfg.test_dataset.scene}/gt.obj')
        evaluate_result = evaluator.evaluate(mesh, mesh_gt)
        print_result(evaluate_result)


def run_evaluate_render():
    from lib.datasets.make_dataset import make_dataset_render
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    from lib.networks import make_network
    from lib.utils.net_utils import load_network
    from lib.evaluators.plane_seg import Evaluator

    network = make_network(cfg).cuda()
    load_network(
        network,
        cfg.trained_model_dir,
        resume=cfg.resume,
        epoch=cfg.test.epoch
    )
    network.eval()
    # data_loader = make_dataset_render(cfg)
    # evaluator = make_evaluator(cfg)
    evaluator = Evaluator()
    render_loader = make_render_data_loader(cfg, is_train=False)

    save_dir = os.path.join(render_loader.dataset.instance_dir, cfg.exp_name)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    eval_render(network, render_loader, 'cuda:0', use_surface_render=True, save_dir=save_dir, cfg=cfg)
    # render_time_test(network, render_loader, 'cuda:0', use_surface_render=True, save_dir=save_dir, cfg=cfg)

    # for plane_mask in plane_masks:
    #     evaluate_result = evaluator.evaluate(plane_mask, mesh_gt)
    # print_result(evaluate_result)


if __name__ == '__main__':
    globals()['run_' + args.type]()
