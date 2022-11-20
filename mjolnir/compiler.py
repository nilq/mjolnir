from __future__ import annotations

import ctypes
import math
from typing import TYPE_CHECKING, Union

import llvmlite.binding as llvm
import numpy as np
from llvmlite import ir
