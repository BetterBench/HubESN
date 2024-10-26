import numpy as np
# import HubESN.esn_utils.utils as utils
# from HubESN.esn_lib.esn_base import *
import esn_utils.utils as utils
from esn_lib.esn_base import *
import mpl_toolkits.axisartist as axisartist


np.random.seed(0)  # 设置随机种子为0
class REHubESN(ESNBase):
    def __init__(self, **kwargs):
        """
        Bio-realistic Hub Echo State Network
        params:
            lambda_dc: distance constraint of connections
            lambda_sc: generation sequence constraint of connections
        """
        super().__init__(**kwargs)

        self.lambda_dc = kwargs.get("lambda_dc", 0.5)
        self.lambda_sc = kwargs.get("lambda_sc", 0.5)
        self.exp_coef = kwargs.get("exp_coef", 2)

        # initialize EI_Balanced_ESN
        self._generate_wrc()
        self._generate_wrc_mask()
        self._generate_wir()
        self._generate_wir_mask()

        self._apply_spec_rad()


    def _generate_distant_constraints(self):
        """
        Generate distant constraints
        """
        np.random.seed(0)  # 固定随机数种子为 0
        x = np.random.normal(0, 1, self.n_size)
        y = np.random.normal(0, 1, self.n_size)
        z = np.random.normal(0, 1, self.n_size)
        self. x, self.y, self.z = x, y, z
        pos = np.array([x, y, z]).T

        # calculate the distance between each node
        dist = np.zeros((self.n_size, self.n_size))
        for i in range(self.n_size):
            for j in range(self.n_size):
                if i != j:
                    dist[i, j] = np.linalg.norm(pos[i] - pos[j])

        mean = np.mean(dist)
        for i in range(self.n_size):
            for j in range(self.n_size):
                if i == j: dist[i, j] = mean

        dist = dist**self.exp_coef
        # dist = (dist - np.min(dist)) / (np.max(dist) - np.min(dist))
        # 使较远距离的权重增大，近距离的权小
        dist = (np.max(dist) - dist) / (np.max(dist) - np.min(dist))
        return dist


    def _generate_sequence_contrainsts(self):
        """
        Generate sequence constraints
        """
        # generate a generate sequence matrix
        seq = np.arange(self.n_size)
        hori_seq = np.tile(seq, (self.n_size, 1))
        vert_seq = np.tile(seq, (self.n_size, 1)).T

        seq_mat = np.abs(hori_seq + vert_seq)**self.exp_coef
        # 正序列约束
        # seq_mat = (seq_mat - np.min(seq_mat)) / (np.max(seq_mat) - np.min(seq_mat))
        # 反序列约束
        seq_mat = (np.max(seq_mat) - seq_mat) / (np.max(seq_mat) - np.min(seq_mat))
        return seq_mat

    def _generate_wrc_mask(self):
        """
        Generate recurrent weights connectivity
        """
        self.dc = self._generate_distant_constraints()
        self.sc = self._generate_sequence_contrainsts()

        # combine constraints and generate deletion probability
        # 度越小，权重越大
        W_prob = self.dc * self.lambda_dc + self.sc * self.lambda_sc
        # 反Hub权重初始化，度越大，权重越大
        # W_prob = (1 - self.dc) * self.lambda_dc + (1 - self.sc) * self.lambda_sc

        self.W_prob = W_prob / W_prob.sum()

        W_rc_mask = np.ones((self.n_size, self.n_size), dtype=int)
        delete_num = int(self.n_size * self.n_size * (1 - self.p2))
        delete_idx = np.random.choice(np.arange(self.n_size * self.n_size), 
                                      size=delete_num, 
                                      p=self.W_prob.reshape(-1), 
                                      replace=False)
        W_rc_mask.reshape(-1)[delete_idx] = 0

        self.not_to_print += ["dc", "sc", "W_prob"]

        self.W_rc_mask = W_rc_mask.reshape(self.n_size, self.n_size)
        self.W_rc = self.W_rc * self.W_rc_mask


    def plot_constrains(self):
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        plt.tight_layout()
        fig.colorbar(axs[0, 0].imshow(self.dc), ax=axs[0, 0])
        fig.colorbar(axs[0, 1].imshow(self.sc), ax=axs[0, 1])
        fig.colorbar(axs[1, 0].imshow(self.W_prob), ax=axs[1, 0])
        fig.colorbar(axs[1, 1].imshow(self.W_rc_mask), ax=axs[1, 1])
        axs[0, 0].set_title("Distant constraints")
        axs[0, 1].set_title("Sequence constraints")
        axs[1, 0].set_title("Deletion probability")
        axs[1, 1].set_title("Deletion mask")
        plt.show()


    def get_save_dict(self):
        specs = self._get_params()
        return specs


    def print_layer(self):
        specs = self.get_save_dict()
        utils.print_params("HubESN", specs)

    def plot_eigenvalues(self):
        eigenvalues, _ = np.linalg.eig(self.W_rc)
        complex_number = np.array(eigenvalues)
        real_parts = [z.real for z in complex_number]
        imaginary_parts = [z.imag for z in complex_number]

        # 画图
        font = {'family': 'Arial',
                'weight': 'normal',
                'size': 12,
                }

        # 画圆
        # 1.圆半径
        r = 1
        # 2.圆心坐标
        a, b = (0., 0.)
        theta = np.arange(0, 2 * np.pi, 0.01)
        x = a + r * np.cos(theta)
        y = b + r * np.sin(theta)

        fig = plt.figure(figsize=(5, 4.5), dpi=600)
        ax = axisartist.Subplot(fig, 111)  # 使用axisartist.Subplot方法创建一个绘图区对象ax
        fig.add_axes(ax)  # 将绘图区对象添加到画布中

        ax.axis[:].set_visible(False)  # 隐藏原来的实线矩形

        ax.axis["x"] = ax.new_floating_axis(0, 0, axis_direction="bottom")  # 添加x轴
        ax.axis["y"] = ax.new_floating_axis(1, 0, axis_direction="bottom")  # 添加y轴

        ax.axis["x"].set_axis_direction('bottom')
        ax.axis["y"].set_axis_direction('left')

        ax.axis["x"].set_axisline_style("->", size=1.5)  # 给x坐标轴加箭头
        ax.axis["y"].set_axisline_style("->", size=1.5)  # 给y坐标轴加箭头

        ax.annotate(xy=(1.3, 0.08), text='Re', fontsize=18)  # 标注x轴
        ax.annotate(xy=(-0.25, 1.3), text='Im', fontsize=18)  # 标注y轴

        plt.xlim(-1.5, 1.5)  # 设置横坐标范围
        plt.ylim(-1.5, 1.5)  # 设置纵坐标范围
        ax.set_xticks([])  # 设置x轴刻度
        ax.set_yticks([])  # 设置y轴刻度

        plt.scatter(real_parts, imaginary_parts, color='#9D26D0', alpha=0.6, s=15)
        plt.plot(x, y, c='black', lw=1.2, linestyle='--')
        plt.tick_params(labelsize=10)
        # plt.savefig(r"C:\Users\Cro\Desktop\IEEE Trans\figures\eigenvalues_esn_1.jpg",bbox_inches='tight')
        plt.show()