# FAMLINN
import re

import torch
import torch.nn as nn
from typing import List

import shutil
import os
import sys
import pathlib


class Container:
    __NEXT_ID = 0

    def __init__(self):
        Container.__NEXT_ID += 1
        self.id = Container.__NEXT_ID
        self.data = None

    def write(self, data): self.data = data

    def read(self): return self.data

    def get_id(self): return self.id

    @staticmethod
    def null_id():
        Container.__NEXT_ID = 0


class Storage:

    def __init__(self):
        self.inputContainer = Container()
        self.outputContainer = self.inputContainer
        self.data = {self.inputContainer.get_id(): self.inputContainer}

    def get_container(self, container_id: int, is_input: bool = False, is_output: bool = False):
        if is_input:
            return self.inputContainer
        if is_output:
            return self.outputContainer
        if container_id not in self.data:
            return None
        return self.data[container_id]

    def add_container(self, set_output: bool = True) -> Container:
        cont = Container()
        self.data[cont.get_id()] = cont
        if set_output:
            self.outputContainer = cont
        return cont

    def input_id(self):
        return self.inputContainer.get_id()

    def output_id(self):
        return self.outputContainer.get_id()

    def pprint(self):
        print("Input container ", self.inputContainer.get_id())
        print("Output container", self.outputContainer.get_id())


class Node:

    def __init__(self, funct, res, args=None, storage: Storage = None, label=""):
        assert funct is not None, "Can not make Node with no function"
        assert storage is not None, "Can not make Node with no storage"
        if args is None:
            args = []

        self.funct = funct
        self.args = args
        self.res = res
        self.storage = storage
        self.label = label

    def save_params(self, path: str):
        torch.save(self.funct.module, path)

    def eval(self, verbose=False):
        arguments = [self.storage.get_container(i).read() for i in self.args]
        #print("EVAL", arguments[0].shape)
        if verbose:
            print(arguments)
        result = self.funct(arguments)
        if verbose:
            print(result)
        self.res.write(result)

    def pprint(self):
        print("Node from {} to {} ({})".format(self.args, self.res.get_id(), self.label))


# Formatted AutoML Iterable Neural Network
class NeuralNetwork:

    def __init__(self):
        Container.null_id()
        self.storage = Storage()
        self.nodes = []

    def add_layer(self, function, input_layers=None, label="") -> int:
        if input_layers is None:
            input_layers = []
        res = self.storage.add_container()
        if len(input_layers) == 0:
            input_layers = [self.storage.input_id()]
        node = Node(function, res, input_layers, self.storage, label=label)
        self.nodes.append(node)
        return res.get_id()

    def insert_layer(self, function, label="", idx=0) -> int:
        res = self.storage.add_container(set_output=False)
        input_layers = []
        if idx != 0:
            input_layers = [self.nodes[idx - 1].res.get_id()]
        node = Node(function, res, input_layers, self.storage, label=label)
        self.nodes.insert(idx, node)
        self.nodes[idx + 1].args = [len(self.nodes) + 1]
        return res.get_id()

    def change_layer(self, idx, function, label=""):
        self.nodes[idx].funct = function
        self.nodes[idx].label = label
        return

    def delete_layer(self, idx):
        res = self.storage.add_container(set_output=False)
        to_del = self.nodes[idx].res.get_id()
        self.nodes[idx + 1].args = self.nodes[idx].args
        self.nodes.pop(idx)
        return

    def add_var(self, data) -> int:
        res = self.storage.add_container(False)
        res.write(data)
        return res.get_id()

    def eval(self, data):
        self.storage.get_container(self.storage.input_id()).write(data)
        for i in self.nodes:
            try:
                i.eval()
            except BaseException as e:
                print("Error in FAMLINN::NeuralNetwork.eval", file=sys.stderr)
                print(str(e), file=sys.stderr)
                i.pprint()  # TODO: REMOVE
                raise e
        return self.storage.get_container(self.storage.output_id()).read()

    def pprint(self):
        print("Print Neural Network:")
        self.storage.pprint()
        for i in self.nodes:
            i.pprint()


class FamlinnTensorOperation(nn.Module):
    pass


class FamlinnAtomicModule(nn.Module):

    def __init__(self, module: nn.Module, source: str):
        super().__init__()
        self.module = module
        self.source = source

    def forward(self, *tensor: torch.Tensor) -> torch.Tensor:
        return self.module(*tensor)

    def __str__(self):
        return self.source

class TorchTensorSqueeze(FamlinnTensorOperation):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, *tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat(tensor, dim=self.dim)

    def __str__(self):
        return "TorchTensorCat(" + str(self.dim) + ")"


class TorchTensorCat(FamlinnTensorOperation):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, *tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat(tensor, dim=self.dim)

    def __str__(self):
        return "TorchTensorCat(" + str(self.dim) + ")"


class TorchTensorSmartReshape(FamlinnTensorOperation):

    def __init__(self):
        super().__init__()

    def forward(self, tensor1: torch.Tensor) -> torch.Tensor:
        return tensor1.reshape(tensor1.size(0), -1)


class TorchTensorFlatten(FamlinnTensorOperation):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.flatten(tensor, self.dim)

    def __str__(self):
        return "TorchTensorFlatten(" + str(self.dim) + ")"


class TorchTensorAdd(FamlinnTensorOperation):

    def __init__(self):
        super().__init__()

    def forward(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
        return tensor1 + tensor2


class TorchTensorTo1D(FamlinnTensorOperation):

    def __init__(self):
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(tensor.size(0), -1)


class TorchTensorSmartView(FamlinnTensorOperation):

    def __init__(self, arg):
        super().__init__()
        self.arg = arg

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(self.arg)

    def __str__(self):
        return "TorchTensorSmartView(" + str(self.arg) + ")"


class TorchTensorVagueView(FamlinnTensorOperation):

    def __init__(self, arg):
        super().__init__()
        self.arg = arg

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(-1, self.arg)

    def __str__(self):
        return "TorchTensorVagueView(" + str(self.arg) + ")"


class Evaluator:
    def __init__(self, module: nn.Module):
        self.module = module

    def __call__(self, *args, **kwargs):
        try:
            return self.module(*args[0])
        except BaseException as e:
            print("Error in FAMLINN::Evaluator", file=sys.stderr)
            print(str(e), file=sys.stderr)
            print(self, file=sys.stderr)
            raise e

    def __str__(self):
        return str(self.module)


class CodeGen:

    def __init__(self, output: pathlib.Path = None):
        self.output = output
        self.result = []

    def add_line(self, line: str):
        self.result.append(line)

    def add_lines(self, lines: List[str]):
        self.result.extend(lines)

    def tabulate_add(self, codegen):
        self.result.extend(list(map(lambda x: "    " + x, codegen.result)))

    def write(self):
        with open(self.output, 'w') as file:
            file.write("\n".join(self.result))


class FAMLINN(NeuralNetwork):

    def __init__(self):
        self._graph_hook_map = {}
        super().__init__()

    def isSampleModule(self, module: nn.Module):
        return (isinstance(module, Evaluator)
                or isinstance(module, nn.Identity)
                or isinstance(module, FamlinnTensorOperation)
                or isinstance(module, FamlinnAtomicModule)
                or isinstance(module, nn.Conv2d)
                or isinstance(module, nn.ConvTranspose2d)
                or isinstance(module, nn.BatchNorm1d)
                or isinstance(module, nn.BatchNorm2d)
                or isinstance(module, nn.Linear)
                or isinstance(module, nn.AvgPool1d)
                or isinstance(module, nn.AvgPool2d)
                or isinstance(module, nn.AvgPool3d)
                or isinstance(module, nn.AdaptiveAvgPool1d)
                or isinstance(module, nn.AdaptiveAvgPool2d)
                or isinstance(module, nn.AdaptiveAvgPool3d)
                or isinstance(module, nn.MaxPool1d)
                or isinstance(module, nn.MaxPool2d)
                or isinstance(module, nn.MaxPool3d)
                or isinstance(module, nn.FractionalMaxPool2d)
                or isinstance(module, nn.FractionalMaxPool3d)
                or isinstance(module, nn.Dropout)
                or isinstance(module, nn.Dropout2d)
                or isinstance(module, nn.ReLU)
                or isinstance(module, nn.LeakyReLU)
                or isinstance(module, nn.Sigmoid)
                or isinstance(module, nn.Tanh)
                or isinstance(module, nn.Flatten)
                )

    def _convert(self, net: nn.Module, argValues, argStorage = None):
        if self.isSampleModule(net):
            self.add_layer(Evaluator(net),
                           [self.storage.output_id()] if argStorage is None else argStorage,
                           str(net))
            return
        trace = torch.jit.trace(net, *argValues, strict=False)
        trace = trace.graph
        nodes_map = {}
        result_map = {}
        name_map = {}
        firstArg = None
        for node in trace.nodes():
            used_variables = re.findall('%[0-9a-zA-Z_.]+', str(node))
            if re.match('.*GetAttr.*', str(node)):
                name_map[used_variables[0][1:]] = re.findall('name=\".*\"', str(node))[0][6:-1]
            if re.match('%[0-9]+', used_variables[0]):
                submodule_name = used_variables[1][1:]
                for idx1, m1 in net.named_children():
                    #print(idx1, name_map[submodule_name], submodule_name, m1.__class__)
                    idx = idx1
                    m = m1
                    if isinstance(m, nn.ModuleList):
                        for idx2, m2 in m.named_children():
                            if idx2 == name_map[submodule_name]:
                                idx = idx2
                                m = m2
                    if idx == name_map[submodule_name]:
                        inps = used_variables[2:]
                        outs = used_variables[0]
                        if firstArg is None:
                            firstArg = inps[0]
                            result_map[firstArg] = argValues[0]
                            nodes_map[firstArg] = self.storage.output_id() if argStorage is None else argStorage[0]
                            argv = argValues
                            args = [nodes_map[firstArg]]
                        else:
                            argv = [result_map[i] for i in inps]
                            args = [nodes_map[i] for i in inps]
                        self._convert(m, argv, args)
                        nodes_map[outs] = self.storage.output_id()
                        result_map[outs] = m(*argv)

    def hook_net(self, net: nn.Module, arg):
        self._convert(net, [arg])
        # self.pprint()

    def export(self, output_src: pathlib.Path, output_weights: pathlib.Path, new_import=[]):
        codegen = CodeGen(output_src)
        self.generate(codegen, new_import)
        codegen.write()
        self.write_weights(output_weights)

    def generate(self, codegen: CodeGen, new_import) -> None:
        codegen.add_lines([
            "from torch.nn import *",
            "from src.famlinn import *",
            "import base64"]
            + new_import + [
            "",
            "",
            "class Net(nn.Module):",
        ])
        codegen.tabulate_add(self.generate_constructor())
        codegen.add_line("")
        codegen.tabulate_add(self.generate_forward())
        codegen.add_line("")
        codegen.tabulate_add(self.generate_read())
        codegen.add_line("")

    def generate_constructor(self) -> CodeGen:
        res_codegen = CodeGen()
        res_codegen.add_lines([
            "def __init__(self):",
            "    super().__init__()",
        ])
        codegen = CodeGen()
        for i, node in enumerate(self.nodes):
            line = "self.{} = {}".format('node_' + str(i), node.label)
            codegen.add_line(line)
        res_codegen.tabulate_add(codegen)
        return res_codegen

    def generate_forward(self) -> CodeGen:
        res_codegen = CodeGen()
        res_codegen.add_lines([
            "def forward(self, arg):",
            "    res_0 = arg",
        ])
        codegen = CodeGen()
        for i, node in enumerate(self.nodes):
            args = '[' + ",".join(map(lambda x: 'res_{}'.format(x - 1), node.args)) + ']'
            line = "res_{} = self.node_{}(*{})  # {}".format(i + 1, i, args, node.label)
            codegen.add_line(line)
        codegen.add_line("return res_{}".format(len(self.nodes)))
        res_codegen.tabulate_add(codegen)
        return res_codegen

    def generate_read(self) -> CodeGen:
        res_codegen = CodeGen()
        res_codegen.add_lines([
            "def read(self, weights_path):",
        ])
        for i, node in enumerate(self.nodes):
            res_codegen.add_lines([
                '        self.node_{} = torch.load(weights_path + \'\\\\node{}\')'.format(i, i)
            ])
        return res_codegen

    def write_weights(self, output_weights: pathlib.Path):
        shutil.rmtree(output_weights, ignore_errors=True)
        os.mkdir(output_weights)
        for i, node in enumerate(self.nodes):
            node.save_params(output_weights / ('node' + str(i)))
