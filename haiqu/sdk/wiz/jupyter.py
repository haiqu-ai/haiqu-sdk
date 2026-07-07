"""
Haiqu SDK.
Wiz (ウィズ) is a supporting library for painting magic.
Jupyter magic happens here.
"""

import re
from collections import defaultdict
from datetime import datetime
from time import sleep
from typing import Union, Iterable
import uuid

import plotly.graph_objects as go

from tabulate import tabulate

try:
    import ipywidgets as widgets
    from IPython.display import display, HTML, Javascript
except ImportError:
    print("Jupyter required! Called function intended to work in Notebook/Lab environment.")

from haiqu.sdk import schemas
from haiqu.sdk.wiz.job_graph import draw_run_job  # noqa: F401

DATE_TIME_FORMAT = "%B %d, %Y %I:%M:%S %p"
DATE_TIME_FORMAT_JOBS = "%b %d, %I:%M %p"

HAIQU_LOGO = """<img height="24" width="24" style="vertical-align: middle;" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB0AAAAdCAMAAABhTZc9AAAA5FBMVEUAAAD////MzMzV1dXb29vd3d3b29vc3Nzb29vc3Nzd3d3W1tbY2Nja2trb29va2trb29vY2NjZ2dnZ2dnY2NjY2NjY2NjZ2dnZ2dna2trY2Nja2tra2trY2NjZ2dnZ2dna2trY2NjY2NjZ2dnY2NjZ2dnZ2dnZ2dnZ2dnZ2dnZ2dnZ2dnZ2dna2trY2NjZ2dnY2NjZ2dnZ2dnZ2dnZ2dna2trY2NjZ2dnZ2dnZ2dna2trZ2dnZ2dnZ2dnZ2dnZ2dnZ2dnZ2dnZ2dnZ2dnZ2dnZ2dnZ2dnZ2dnZ2dnZ2dnZ2dnZ2dkLVpmKAAAAS3RSTlMAAQUGDg8VFhwdHh8hIiMpKjs8PVVWW11fYWJ7fH1+gYKDnp+lpqeoqa2ur7CxsrO5uru8y8zNz9DS09vc3t/g5OXm6+zt8fj5/P1AfbQ4AAAAAWJLR0QB/wIt3gAAAVZJREFUKM+1U11XgkAUXFGRTAtNRQyT/ADdDTNTU5BMUQju//8/sbgi0LM87L1z5tyzszMXhG7+1Qdbd9ltvqxca1DPcEUdfgj2AcCfkD1oxSQpmE6LQ0gP2RFCnOSYQmLS/Lqj9TPYBR+0Ka/NQszqTpmWh18ZyV4lop3RhXyCFi2590l4YiNHgQQXaYM9R8urzYcnb6sUcIc+Y7eETqonMULiSaXTbxZjXYxQdWo/MijaRjVkXQaX/njuTfhYJI+9+dhfMKQABHLKHDkAeGZ9E+A7490OoMHa7r/ZTjirsH7lazMPX+8tYW+mxfdSzRXDFmPNpJLQnH1vL/Ve5pUaeVWyexTkY6/qIEU+TzH1mUQ+t6F2kaGdM6p6MuqcM7o/DhP5riN6HuY7O+e7KaR2Q7ruRr593AipvRrBwSAB3StiHGBYyO5k33IXSkNZuFa/dvtf4A+G0zEe41ZDhgAAAABJRU5ErkJggg==" />"""  # noqa: E501

TEMPLATE_WIDGET = """
<style>
table {{width: 100%}}
td {{text-align: left !important}}
div.haiqu_widget {{bacground-color:var(--jp-cell-editor-background);min-height: {};color: var(--jp-content-font-color2);border: 1px solid rgba(70, 70, 70, 1);border-radius: 16px;padding:15px;}}
div.haiqu_widget_tile {{float: left; width: 45%; margin-right: 10px; margin-bottom: 10px; min-height: 620px;}}
div.haiqu_widget_tile_sm {{float: left; width: 45%; margin-right: 10px; margin-bottom: 10px; min-height: 520px;}}
p.haiqu_widget_title {{text-transform:uppercase;font-size:16px;font-weight: 500;margin-top:0px!important}}
p.haiqu_widget_title span {{vertical-align: middle;}}
p.haiqu_widget_title_bottom {{border: 1px solid rgba(34, 34, 34, 1)}}
div.haiqu_widget pre {{background: none!important;}}
.vectors_bg {{
    font-color: var(--jp-content-font-color2);
    background-repeat: no-repeat;
    background-position: top right;
    background-image: linear-gradient(to right, var(--jp-cell-editor-background) 50%,
        color-mix(in srgb, var(--jp-cell-editor-active-background) 85%, transparent)), 
        url("data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzU0IiBoZWlnaHQ9IjU1IiB2aWV3Qm94PSIwIDAgMzU0IDU1IiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8ZyBmaWx0ZXI9InVybCgjZmlsdGVyMF9mXzIzMTNfNTApIj4KPHBhdGggZD0iTTEwMi42NjIgMTE3LjAyNEwxMDAuNTQxIDkyLjk4MjVDOTkuNDM0NyA4MC40NDY3IDEwOS45MjUgNjkuOTU2MSAxMjIuNDYxIDcxLjA2MjJWNzEuMDYyMkMxMzQuOTk3IDcyLjE2ODMgMTQ1LjQ4OCA2MS42Nzc4IDE0NC4zODEgNDkuMTQxOVY0OS4xNDE5QzE0My4yNzUgMzYuNjA2IDE1My43NjYgMjYuMTE1NSAxNjYuMzAyIDI3LjIyMTZWMjcuMjIxNkMxNzguODM4IDI4LjMyNzcgMTg5LjMyOCAxNy44MzcyIDE4OC4yMjIgNS4zMDEyOVY1LjMwMTI5QzE4Ny4xMTYgLTcuMjM0NiAxOTcuNjA2IC0xNy43MjUxIDIxMC4xNDIgLTE2LjYxOVYtMTYuNjE5QzIyMi42NzggLTE1LjUxMjkgMjMzLjE2OSAtMjYuMDAzNSAyMzIuMDYzIC0zOC41MzkzVi0zOC41MzkzQzIzMC45NTcgLTUxLjA3NTIgMjQxLjQ0NyAtNjEuNTY1OCAyNTMuOTgzIC02MC40NTk2TDI3OC4wMjUgLTU4LjMzODMiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIvPgo8L2c+CjxnIGZpbHRlcj0idXJsKCNmaWx0ZXIxX2ZfMjMxM181MCkiPgo8cGF0aCBkPSJNMjM5LjI5MyAxNjUuNjU1TDIzNy4xNzIgMTQxLjYxM0MyMzYuMDY2IDEyOS4wNzcgMjQ2LjU1NiAxMTguNTg3IDI1OS4wOTIgMTE5LjY5M1YxMTkuNjkzQzI3MS42MjggMTIwLjc5OSAyODIuMTE5IDExMC4zMDkgMjgxLjAxMiA5Ny43NzI2Vjk3Ljc3MjZDMjc5LjkwNiA4NS4yMzY4IDI5MC4zOTcgNzQuNzQ2MiAzMDIuOTMzIDc1Ljg1MjNWNzUuODUyM0MzMTUuNDY5IDc2Ljk1ODQgMzI1Ljk1OSA2Ni40Njc5IDMyNC44NTMgNTMuOTMyVjUzLjkzMkMzMjMuNzQ3IDQxLjM5NjEgMzM0LjIzNyAzMC45MDU2IDM0Ni43NzMgMzIuMDExN1YzMi4wMTE3QzM1OS4zMDkgMzMuMTE3OCAzNjkuOCAyMi42MjczIDM2OC42OTQgMTAuMDkxNFYxMC4wOTE0QzM2Ny41ODggLTIuNDQ0NDggMzc4LjA3OCAtMTIuOTM1IDM5MC42MTQgLTExLjgyODlMNDE0LjY1NiAtOS43MDc1OCIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIi8+CjwvZz4KPGcgZmlsdGVyPSJ1cmwoI2ZpbHRlcjJfZl8yMzEzXzUwKSI+CjxwYXRoIGQ9Ik0yMS4zNDQ1IDEyNC44MDJMMTkuMjIzMSAxMDAuNzZDMTguMTE3IDg4LjIyNDUgMjguNjA3NiA3Ny43MzQgNDEuMTQzNSA3OC44NDAxVjc4Ljg0MDFDNTMuNjc5MyA3OS45NDYyIDY0LjE2OTkgNjkuNDU1NyA2My4wNjM4IDU2LjkxOThWNTYuOTE5OEM2MS45NTc3IDQ0LjM4MzkgNzIuNDQ4MiAzMy44OTM0IDg0Ljk4NDEgMzQuOTk5NVYzNC45OTk1Qzk3LjUyIDM2LjEwNTYgMTA4LjAxIDI1LjYxNTEgMTA2LjkwNCAxMy4wNzkyVjEzLjA3OTJDMTA1Ljc5OCAwLjU0MzI5OCAxMTYuMjg5IC05Ljk0NzI0IDEyOC44MjUgLTguODQxMTNWLTguODQxMTNDMTQxLjM2MSAtNy43MzUwMiAxNTEuODUxIC0xOC4yMjU2IDE1MC43NDUgLTMwLjc2MTRWLTMwLjc2MTRDMTQ5LjYzOSAtNDMuMjk3MyAxNjAuMTI5IC01My43ODc5IDE3Mi42NjUgLTUyLjY4MThMMTk2LjcwNyAtNTAuNTYwNCIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIi8+CjwvZz4KPGcgZmlsdGVyPSJ1cmwoI2ZpbHRlcjNfZl8yMzEzXzUwKSI+CjxwYXRoIGQ9Ik0tMTAwLjY1NSAxMzYuODAyTC0xMDIuNzc3IDExMi43NkMtMTAzLjg4MyAxMDAuMjI1IC05My4zOTIyIDg5LjczNDEgLTgwLjg1NjQgOTAuODQwMlY5MC44NDAyQy02OC4zMjA1IDkxLjk0NjMgLTU3LjgyOTkgODEuNDU1NyAtNTguOTM2IDY4LjkxOTlWNjguOTE5OUMtNjAuMDQyMiA1Ni4zODQgLTQ5LjU1MTYgNDUuODkzNCAtMzcuMDE1NyA0Ni45OTk2VjQ2Ljk5OTZDLTI0LjQ3OTkgNDguMTA1NyAtMTMuOTg5MyAzNy42MTUxIC0xNS4wOTU0IDI1LjA3OTJWMjUuMDc5MkMtMTYuMjAxNSAxMi41NDM0IC01LjcxMSAyLjA1MjgyIDYuODI0ODggMy4xNTg5M1YzLjE1ODkzQzE5LjM2MDggNC4yNjUwNCAyOS44NTEzIC02LjIyNTUgMjguNzQ1MiAtMTguNzYxNFYtMTguNzYxNEMyNy42MzkxIC0zMS4yOTczIDM4LjEyOTYgLTQxLjc4NzggNTAuNjY1NSAtNDAuNjgxN0w3NC43MDcxIC0zOC41NjA0IiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiLz4KPC9nPgo8ZyBmaWx0ZXI9InVybCgjZmlsdGVyNF9mXzIzMTNfNTApIj4KPHBhdGggZD0iTTE1OC41MjMgMTcyLjg4NUwxNTYuNDAyIDE0OC44NDRDMTU1LjI5NiAxMzYuMzA4IDE2NS43ODcgMTI1LjgxNyAxNzguMzIyIDEyNi45MjRWMTI2LjkyNEMxOTAuODU4IDEyOC4wMyAyMDEuMzQ5IDExNy41MzkgMjAwLjI0MyAxMDUuMDAzVjEwNS4wMDNDMTk5LjEzNyA5Mi40NjczIDIwOS42MjcgODEuOTc2OCAyMjIuMTYzIDgzLjA4MjlWODMuMDgyOUMyMzQuNjk5IDg0LjE4OSAyNDUuMTg5IDczLjY5ODUgMjQ0LjA4MyA2MS4xNjI2VjYxLjE2MjZDMjQyLjk3NyA0OC42MjY3IDI1My40NjggMzguMTM2MiAyNjYuMDA0IDM5LjI0MjNWMzkuMjQyM0MyNzguNTQgNDAuMzQ4NCAyODkuMDMgMjkuODU3OSAyODcuOTI0IDE3LjMyMlYxNy4zMjJDMjg2LjgxOCA0Ljc4NjEgMjk3LjMwOCAtNS43MDQ0NCAzMDkuODQ0IC00LjU5ODMzTDMzMy44ODYgLTIuNDc3MDEiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIvPgo8L2c+CjxnIGZpbHRlcj0idXJsKCNmaWx0ZXI1X2ZfMjMxM181MCkiPgo8cGF0aCBkPSJNNTguMTE0NSA3Mi40NzU2TDgyLjE1NjEgNzQuNTk2OUM5NC42OTIgNzUuNzAzIDEwNS4xODMgNjUuMjEyNSAxMDQuMDc2IDUyLjY3NjZWNTIuNjc2NkMxMDIuOTcgNDAuMTQwNyAxMTMuNDYxIDI5LjY1MDIgMTI1Ljk5NyAzMC43NTYzVjMwLjc1NjNDMTM4LjUzMyAzMS44NjI0IDE0OS4wMjMgMjEuMzcxOSAxNDcuOTE3IDguODM1OTlWOC44MzU5OUMxNDYuODExIC0zLjY5OTg5IDE1Ny4zMDIgLTE0LjE5MDQgMTY5LjgzNyAtMTMuMDg0M1YtMTMuMDg0M0MxODIuMzczIC0xMS45NzgyIDE5Mi44NjQgLTIyLjQ2ODcgMTkxLjc1OCAtMzUuMDA0NlYtMzUuMDA0NkMxOTAuNjUyIC00Ny41NDA1IDIwMS4xNDIgLTU4LjAzMSAyMTMuNjc4IC01Ni45MjQ5Vi01Ni45MjQ5QzIyNi4yMTQgLTU1LjgxODggMjM2LjcwNCAtNjYuMzA5NCAyMzUuNTk4IC03OC44NDUyTDIzMy40NzcgLTEwMi44ODciIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIvPgo8L2c+CjxnIGZpbHRlcj0idXJsKCNmaWx0ZXI2X2ZfMjMxM181MCkiPgo8cGF0aCBkPSJNLTYzLjg4NTUgODQuNDc1NUwtMzkuODQzOSA4Ni41OTY4Qy0yNy4zMDggODcuNzAyOSAtMTYuODE3NCA3Ny4yMTI0IC0xNy45MjM1IDY0LjY3NjVWNjQuNjc2NUMtMTkuMDI5NyA1Mi4xNDA2IC04LjUzOTEyIDQxLjY1MDEgMy45OTY3NyA0Mi43NTYyVjQyLjc1NjJDMTYuNTMyNiA0My44NjIzIDI3LjAyMzIgMzMuMzcxOCAyNS45MTcxIDIwLjgzNTlWMjAuODM1OUMyNC44MTEgOC4yOTk5OSAzNS4zMDE1IC0yLjE5MDU1IDQ3LjgzNzQgLTEuMDg0NDRWLTEuMDg0NDRDNjAuMzczMyAwLjAyMTY2NTUgNzAuODYzOCAtMTAuNDY4OSA2OS43NTc3IC0yMy4wMDQ4Vi0yMy4wMDQ4QzY4LjY1MTYgLTM1LjU0MDYgNzkuMTQyMSAtNDYuMDMxMiA5MS42NzggLTQ0LjkyNTFWLTQ0LjkyNTFDMTA0LjIxNCAtNDMuODE5IDExNC43MDQgLTU0LjMwOTUgMTEzLjU5OCAtNjYuODQ1NEwxMTEuNDc3IC05MC44ODciIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIvPgo8L2c+CjxnIGZpbHRlcj0idXJsKCNmaWx0ZXI3X2ZfMjMxM181MCkiPgo8cGF0aCBkPSJNMTUwLjc0NSAxNjUuMTA4TDE3NC43ODcgMTY3LjIzQzE4Ny4zMjMgMTY4LjMzNiAxOTcuODEzIDE1Ny44NDUgMTk2LjcwNyAxNDUuMzA5VjE0NS4zMDlDMTk1LjYwMSAxMzIuNzc0IDIwNi4wOTIgMTIyLjI4MyAyMTguNjI4IDEyMy4zODlWMTIzLjM4OUMyMzEuMTY0IDEyNC40OTUgMjQxLjY1NCAxMTQuMDA1IDI0MC41NDggMTAxLjQ2OVYxMDEuNDY5QzIzOS40NDIgODguOTMyOSAyNDkuOTMyIDc4LjQ0MjQgMjYyLjQ2OCA3OS41NDg1Vjc5LjU0ODVDMjc1LjAwNCA4MC42NTQ2IDI4NS40OTUgNzAuMTY0MSAyODQuMzg5IDU3LjYyODJWNTcuNjI4MkMyODMuMjgzIDQ1LjA5MjMgMjkzLjc3MyAzNC42MDE4IDMwNi4zMDkgMzUuNzA3OVYzNS43MDc5QzMxOC44NDUgMzYuODE0IDMyOS4zMzUgMjYuMzIzNSAzMjguMjI5IDEzLjc4NzZMMzI2LjEwOCAtMTAuMjU0MSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIi8+CjwvZz4KPGcgZmlsdGVyPSJ1cmwoI2ZpbHRlcjhfZl8yMzEzXzUwKSI+CjxwYXRoIGQ9Ik0tMjMuOTA5OCA4MC45NjA5TDAuMTMxNzgyIDgzLjA4MjJDMTIuNjY3NyA4NC4xODgzIDIzLjE1ODIgNzMuNjk3OCAyMi4wNTIxIDYxLjE2MTlWNjEuMTYxOUMyMC45NDYgNDguNjI2IDMxLjQzNjUgMzguMTM1NSA0My45NzI0IDM5LjI0MTZWMzkuMjQxNkM1Ni41MDgzIDQwLjM0NzcgNjYuOTk4OCAyOS44NTcyIDY1Ljg5MjcgMTcuMzIxM1YxNy4zMjEzQzY0Ljc4NjYgNC43ODU0IDc1LjI3NzEgLTUuNzA1MTQgODcuODEzIC00LjU5OTAzVi00LjU5OTAzQzEwMC4zNDkgLTMuNDkyOTIgMTEwLjgzOSAtMTMuOTgzNSAxMDkuNzMzIC0yNi41MTkzVi0yNi41MTkzQzEwOC42MjcgLTM5LjA1NTIgMTE5LjExOCAtNDkuNTQ1OCAxMzEuNjU0IC00OC40Mzk2Vi00OC40Mzk2QzE0NC4xOSAtNDcuMzMzNSAxNTQuNjggLTU3LjgyNDEgMTUzLjU3NCAtNzAuMzU5OUwxNTEuNDUzIC05NC40MDE2IiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiLz4KPC9nPgo8ZyBmaWx0ZXI9InVybCgjZmlsdGVyOV9mXzIzMTNfNTApIj4KPHBhdGggZD0iTS0xNDUuOTEgOTIuOTYwOEwtMTIxLjg2OCA5NS4wODIxQy0xMDkuMzMyIDk2LjE4ODMgLTk4Ljg0MTYgODUuNjk3NyAtOTkuOTQ3NyA3My4xNjE4VjczLjE2MThDLTEwMS4wNTQgNjAuNjI2IC05MC41NjMzIDUwLjEzNTQgLTc4LjAyNzQgNTEuMjQxNVY1MS4yNDE1Qy02NS40OTE1IDUyLjM0NzYgLTU1LjAwMSA0MS44NTcxIC01Ni4xMDcxIDI5LjMyMTJWMjkuMzIxMkMtNTcuMjEzMiAxNi43ODUzIC00Ni43MjI3IDYuMjk0OCAtMzQuMTg2OCA3LjQwMDkxVjcuNDAwOTFDLTIxLjY1MDkgOC41MDcwMiAtMTEuMTYwNCAtMS45ODM1MiAtMTIuMjY2NSAtMTQuNTE5NFYtMTQuNTE5NEMtMTMuMzcyNiAtMjcuMDU1MyAtMi44ODIwNSAtMzcuNTQ1OCA5LjY1MzgzIC0zNi40Mzk3Vi0zNi40Mzk3QzIyLjE4OTcgLTM1LjMzMzYgMzIuNjgwMyAtNDUuODI0MSAzMS41NzQxIC01OC4zNkwyOS40NTI4IC04Mi40MDE2IiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiLz4KPC9nPgo8ZyBmaWx0ZXI9InVybCgjZmlsdGVyMTBfZl8yMzEzXzUwKSI+CjxwYXRoIGQ9Ik0xMTMuOTc2IDEyOC4zMzlMMTM4LjAxNyAxMzAuNDZDMTUwLjU1MyAxMzEuNTY2IDE2MS4wNDQgMTIxLjA3NiAxNTkuOTM4IDEwOC41NFYxMDguNTRDMTU4LjgzMiA5Ni4wMDQgMTY5LjMyMiA4NS41MTM1IDE4MS44NTggODYuNjE5NlY4Ni42MTk2QzE5NC4zOTQgODcuNzI1NyAyMDQuODg1IDc3LjIzNTIgMjAzLjc3OCA2NC42OTkzVjY0LjY5OTNDMjAyLjY3MiA1Mi4xNjM0IDIxMy4xNjMgNDEuNjcyOSAyMjUuNjk5IDQyLjc3OVY0Mi43NzlDMjM4LjIzNSA0My44ODUxIDI0OC43MjUgMzMuMzk0NSAyNDcuNjE5IDIwLjg1ODdWMjAuODU4N0MyNDYuNTEzIDguMzIyNzcgMjU3LjAwMyAtMi4xNjc3NyAyNjkuNTM5IC0xLjA2MTY2Vi0xLjA2MTY2QzI4Mi4wNzUgMC4wNDQ0NDkxIDI5Mi41NjYgLTEwLjQ0NjEgMjkxLjQ2IC0yMi45ODJMMjg5LjMzOCAtNDcuMDIzNiIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIi8+CjwvZz4KPGcgZmlsdGVyPSJ1cmwoI2ZpbHRlcjExX2ZfMjMxM181MCkiPgo8cGF0aCBkPSJNODguODcyOCAxMzkuMDY5TDMzNS42NTMgLTEwNy43MTEiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIvPgo8L2c+CjxkZWZzPgo8ZmlsdGVyIGlkPSJmaWx0ZXIwX2ZfMjMxM181MCIgeD0iOTUuNDYwNSIgeT0iLTY1LjUzOTkiIHdpZHRoPSIxODYuNjUyIiBoZWlnaHQ9IjE4Ni42NTIiIGZpbHRlclVuaXRzPSJ1c2VyU3BhY2VPblVzZSIgY29sb3ItaW50ZXJwb2xhdGlvbi1maWx0ZXJzPSJzUkdCIj4KPGZlRmxvb2QgZmxvb2Qtb3BhY2l0eT0iMCIgcmVzdWx0PSJCYWNrZ3JvdW5kSW1hZ2VGaXgiLz4KPGZlQmxlbmQgbW9kZT0ibm9ybWFsIiBpbj0iU291cmNlR3JhcGhpYyIgaW4yPSJCYWNrZ3JvdW5kSW1hZ2VGaXgiIHJlc3VsdD0ic2hhcGUiLz4KPGZlR2F1c3NpYW5CbHVyIHN0ZERldmlhdGlvbj0iMiIgcmVzdWx0PSJlZmZlY3QxX2ZvcmVncm91bmRCbHVyXzIzMTNfNTAiLz4KPC9maWx0ZXI+CjxmaWx0ZXIgaWQ9ImZpbHRlcjFfZl8yMzEzXzUwIiB4PSIyMzIuMDkxIiB5PSItMTYuOTA5MiIgd2lkdGg9IjE4Ni42NTIiIGhlaWdodD0iMTg2LjY1MiIgZmlsdGVyVW5pdHM9InVzZXJTcGFjZU9uVXNlIiBjb2xvci1pbnRlcnBvbGF0aW9uLWZpbHRlcnM9InNSR0IiPgo8ZmVGbG9vZCBmbG9vZC1vcGFjaXR5PSIwIiByZXN1bHQ9IkJhY2tncm91bmRJbWFnZUZpeCIvPgo8ZmVCbGVuZCBtb2RlPSJub3JtYWwiIGluPSJTb3VyY2VHcmFwaGljIiBpbjI9IkJhY2tncm91bmRJbWFnZUZpeCIgcmVzdWx0PSJzaGFwZSIvPgo8ZmVHYXVzc2lhbkJsdXIgc3RkRGV2aWF0aW9uPSIyIiByZXN1bHQ9ImVmZmVjdDFfZm9yZWdyb3VuZEJsdXJfMjMxM181MCIvPgo8L2ZpbHRlcj4KPGZpbHRlciBpZD0iZmlsdGVyMl9mXzIzMTNfNTAiIHg9IjE0LjE0MjkiIHk9Ii01Ny43NjIiIHdpZHRoPSIxODYuNjUyIiBoZWlnaHQ9IjE4Ni42NTIiIGZpbHRlclVuaXRzPSJ1c2VyU3BhY2VPblVzZSIgY29sb3ItaW50ZXJwb2xhdGlvbi1maWx0ZXJzPSJzUkdCIj4KPGZlRmxvb2QgZmxvb2Qtb3BhY2l0eT0iMCIgcmVzdWx0PSJCYWNrZ3JvdW5kSW1hZ2VGaXgiLz4KPGZlQmxlbmQgbW9kZT0ibm9ybWFsIiBpbj0iU291cmNlR3JhcGhpYyIgaW4yPSJCYWNrZ3JvdW5kSW1hZ2VGaXgiIHJlc3VsdD0ic2hhcGUiLz4KPGZlR2F1c3NpYW5CbHVyIHN0ZERldmlhdGlvbj0iMiIgcmVzdWx0PSJlZmZlY3QxX2ZvcmVncm91bmRCbHVyXzIzMTNfNTAiLz4KPC9maWx0ZXI+CjxmaWx0ZXIgaWQ9ImZpbHRlcjNfZl8yMzEzXzUwIiB4PSItMTA3Ljg1NyIgeT0iLTQ1Ljc2MiIgd2lkdGg9IjE4Ni42NTIiIGhlaWdodD0iMTg2LjY1MiIgZmlsdGVyVW5pdHM9InVzZXJTcGFjZU9uVXNlIiBjb2xvci1pbnRlcnBvbGF0aW9uLWZpbHRlcnM9InNSR0IiPgo8ZmVGbG9vZCBmbG9vZC1vcGFjaXR5PSIwIiByZXN1bHQ9IkJhY2tncm91bmRJbWFnZUZpeCIvPgo8ZmVCbGVuZCBtb2RlPSJub3JtYWwiIGluPSJTb3VyY2VHcmFwaGljIiBpbjI9IkJhY2tncm91bmRJbWFnZUZpeCIgcmVzdWx0PSJzaGFwZSIvPgo8ZmVHYXVzc2lhbkJsdXIgc3RkRGV2aWF0aW9uPSIyIiByZXN1bHQ9ImVmZmVjdDFfZm9yZWdyb3VuZEJsdXJfMjMxM181MCIvPgo8L2ZpbHRlcj4KPGZpbHRlciBpZD0iZmlsdGVyNF9mXzIzMTNfNTAiIHg9IjE1MS4zMjIiIHk9Ii05LjY3ODYyIiB3aWR0aD0iMTg2LjY1MiIgaGVpZ2h0PSIxODYuNjUyIiBmaWx0ZXJVbml0cz0idXNlclNwYWNlT25Vc2UiIGNvbG9yLWludGVycG9sYXRpb24tZmlsdGVycz0ic1JHQiI+CjxmZUZsb29kIGZsb29kLW9wYWNpdHk9IjAiIHJlc3VsdD0iQmFja2dyb3VuZEltYWdlRml4Ii8+CjxmZUJsZW5kIG1vZGU9Im5vcm1hbCIgaW49IlNvdXJjZUdyYXBoaWMiIGluMj0iQmFja2dyb3VuZEltYWdlRml4IiByZXN1bHQ9InNoYXBlIi8+CjxmZUdhdXNzaWFuQmx1ciBzdGREZXZpYXRpb249IjIiIHJlc3VsdD0iZWZmZWN0MV9mb3JlZ3JvdW5kQmx1cl8yMzEzXzUwIi8+CjwvZmlsdGVyPgo8ZmlsdGVyIGlkPSJmaWx0ZXI1X2ZfMjMxM181MCIgeD0iNTQuMDI2NiIgeT0iLTEwNi45NzUiIHdpZHRoPSIxODYuNjUyIiBoZWlnaHQ9IjE4Ni42NTIiIGZpbHRlclVuaXRzPSJ1c2VyU3BhY2VPblVzZSIgY29sb3ItaW50ZXJwb2xhdGlvbi1maWx0ZXJzPSJzUkdCIj4KPGZlRmxvb2QgZmxvb2Qtb3BhY2l0eT0iMCIgcmVzdWx0PSJCYWNrZ3JvdW5kSW1hZ2VGaXgiLz4KPGZlQmxlbmQgbW9kZT0ibm9ybWFsIiBpbj0iU291cmNlR3JhcGhpYyIgaW4yPSJCYWNrZ3JvdW5kSW1hZ2VGaXgiIHJlc3VsdD0ic2hhcGUiLz4KPGZlR2F1c3NpYW5CbHVyIHN0ZERldmlhdGlvbj0iMiIgcmVzdWx0PSJlZmZlY3QxX2ZvcmVncm91bmRCbHVyXzIzMTNfNTAiLz4KPC9maWx0ZXI+CjxmaWx0ZXIgaWQ9ImZpbHRlcjZfZl8yMzEzXzUwIiB4PSItNjcuOTczNCIgeT0iLTk0Ljk3NDkiIHdpZHRoPSIxODYuNjUyIiBoZWlnaHQ9IjE4Ni42NTIiIGZpbHRlclVuaXRzPSJ1c2VyU3BhY2VPblVzZSIgY29sb3ItaW50ZXJwb2xhdGlvbi1maWx0ZXJzPSJzUkdCIj4KPGZlRmxvb2QgZmxvb2Qtb3BhY2l0eT0iMCIgcmVzdWx0PSJCYWNrZ3JvdW5kSW1hZ2VGaXgiLz4KPGZlQmxlbmQgbW9kZT0ibm9ybWFsIiBpbj0iU291cmNlR3JhcGhpYyIgaW4yPSJCYWNrZ3JvdW5kSW1hZ2VGaXgiIHJlc3VsdD0ic2hhcGUiLz4KPGZlR2F1c3NpYW5CbHVyIHN0ZERldmlhdGlvbj0iMiIgcmVzdWx0PSJlZmZlY3QxX2ZvcmVncm91bmRCbHVyXzIzMTNfNTAiLz4KPC9maWx0ZXI+CjxmaWx0ZXIgaWQ9ImZpbHRlcjdfZl8yMzEzXzUwIiB4PSIxNDYuNjU4IiB5PSItMTQuMzQxOSIgd2lkdGg9IjE4Ni42NTIiIGhlaWdodD0iMTg2LjY1MiIgZmlsdGVyVW5pdHM9InVzZXJTcGFjZU9uVXNlIiBjb2xvci1pbnRlcnBvbGF0aW9uLWZpbHRlcnM9InNSR0IiPgo8ZmVGbG9vZCBmbG9vZC1vcGFjaXR5PSIwIiByZXN1bHQ9IkJhY2tncm91bmRJbWFnZUZpeCIvPgo8ZmVCbGVuZCBtb2RlPSJub3JtYWwiIGluPSJTb3VyY2VHcmFwaGljIiBpbjI9IkJhY2tncm91bmRJbWFnZUZpeCIgcmVzdWx0PSJzaGFwZSIvPgo8ZmVHYXVzc2lhbkJsdXIgc3RkRGV2aWF0aW9uPSIyIiByZXN1bHQ9ImVmZmVjdDFfZm9yZWdyb3VuZEJsdXJfMjMxM181MCIvPgo8L2ZpbHRlcj4KPGZpbHRlciBpZD0iZmlsdGVyOF9mXzIzMTNfNTAiIHg9Ii0yNy45OTc3IiB5PSItOTguNDg5NSIgd2lkdGg9IjE4Ni42NTIiIGhlaWdodD0iMTg2LjY1MiIgZmlsdGVyVW5pdHM9InVzZXJTcGFjZU9uVXNlIiBjb2xvci1pbnRlcnBvbGF0aW9uLWZpbHRlcnM9InNSR0IiPgo8ZmVGbG9vZCBmbG9vZC1vcGFjaXR5PSIwIiByZXN1bHQ9IkJhY2tncm91bmRJbWFnZUZpeCIvPgo8ZmVCbGVuZCBtb2RlPSJub3JtYWwiIGluPSJTb3VyY2VHcmFwaGljIiBpbjI9IkJhY2tncm91bmRJbWFnZUZpeCIgcmVzdWx0PSJzaGFwZSIvPgo8ZmVHYXVzc2lhbkJsdXIgc3RkRGV2aWF0aW9uPSIyIiByZXN1bHQ9ImVmZmVjdDFfZm9yZWdyb3VuZEJsdXJfMjMxM181MCIvPgo8L2ZpbHRlcj4KPGZpbHRlciBpZD0iZmlsdGVyOV9mXzIzMTNfNTAiIHg9Ii0xNDkuOTk4IiB5PSItODYuNDg5NSIgd2lkdGg9IjE4Ni42NTIiIGhlaWdodD0iMTg2LjY1MiIgZmlsdGVyVW5pdHM9InVzZXJTcGFjZU9uVXNlIiBjb2xvci1pbnRlcnBvbGF0aW9uLWZpbHRlcnM9InNSR0IiPgo8ZmVGbG9vZCBmbG9vZC1vcGFjaXR5PSIwIiByZXN1bHQ9IkJhY2tncm91bmRJbWFnZUZpeCIvPgo8ZmVCbGVuZCBtb2RlPSJub3JtYWwiIGluPSJTb3VyY2VHcmFwaGljIiBpbjI9IkJhY2tncm91bmRJbWFnZUZpeCIgcmVzdWx0PSJzaGFwZSIvPgo8ZmVHYXVzc2lhbkJsdXIgc3RkRGV2aWF0aW9uPSIyIiByZXN1bHQ9ImVmZmVjdDFfZm9yZWdyb3VuZEJsdXJfMjMxM181MCIvPgo8L2ZpbHRlcj4KPGZpbHRlciBpZD0iZmlsdGVyMTBfZl8yMzEzXzUwIiB4PSIxMDkuODg4IiB5PSItNTEuMTExNSIgd2lkdGg9IjE4Ni42NTIiIGhlaWdodD0iMTg2LjY1MiIgZmlsdGVyVW5pdHM9InVzZXJTcGFjZU9uVXNlIiBjb2xvci1pbnRlcnBvbGF0aW9uLWZpbHRlcnM9InNSR0IiPgo8ZmVGbG9vZCBmbG9vZC1vcGFjaXR5PSIwIiByZXN1bHQ9IkJhY2tncm91bmRJbWFnZUZpeCIvPgo8ZmVCbGVuZCBtb2RlPSJub3JtYWwiIGluPSJTb3VyY2VHcmFwaGljIiBpbjI9IkJhY2tncm91bmRJbWFnZUZpeCIgcmVzdWx0PSJzaGFwZSIvPgo8ZmVHYXVzc2lhbkJsdXIgc3RkRGV2aWF0aW9uPSIyIiByZXN1bHQ9ImVmZmVjdDFfZm9yZWdyb3VuZEJsdXJfMjMxM181MCIvPgo8L2ZpbHRlcj4KPGZpbHRlciBpZD0iZmlsdGVyMTFfZl8yMzEzXzUwIiB4PSI4NC4xNjU3IiB5PSItMTEyLjQxOSIgd2lkdGg9IjI1Ni4xOTQiIGhlaWdodD0iMjU2LjE5NCIgZmlsdGVyVW5pdHM9InVzZXJTcGFjZU9uVXNlIiBjb2xvci1pbnRlcnBvbGF0aW9uLWZpbHRlcnM9InNSR0IiPgo8ZmVGbG9vZCBmbG9vZC1vcGFjaXR5PSIwIiByZXN1bHQ9IkJhY2tncm91bmRJbWFnZUZpeCIvPgo8ZmVCbGVuZCBtb2RlPSJub3JtYWwiIGluPSJTb3VyY2VHcmFwaGljIiBpbjI9IkJhY2tncm91bmRJbWFnZUZpeCIgcmVzdWx0PSJzaGFwZSIvPgo8ZmVHYXVzc2lhbkJsdXIgc3RkRGV2aWF0aW9uPSIyIiByZXN1bHQ9ImVmZmVjdDFfZm9yZWdyb3VuZEJsdXJfMjMxM181MCIvPgo8L2ZpbHRlcj4KPC9kZWZzPgo8L3N2Zz4K")
}}
.image_bg {{
}}
.neonQASM {{
    font-family: monospace!important;
    line-height: 0.8rem!important;
    background: none!important;
    border: none!important;
    overflow: scroll!important;
    overflow-y: hidden!important;
    font-size: 0.6rem!important;
    text-wrap: nowrap!important;
    color: var(--jp-content-font-color2);
}}
.neon {{
    color: #FED128!important;
    text-shadow: 0 0 1vw #FA1C16, 0 0 3vw #FA1C16, 0 0 10vw #FA1C16, 0 0 10vw #FA1C16, 0 0 .4vw #FED128, .2vw .2vw .1vw #806914!important;
}}
.neutral {{
    color: #d9d9d9!important;
}}
.haiqu_orange {{
    color: #F27E3D!important;
    text-shadow: 0 0 3vw #F40A35!important;
}}
.haiqu_blue {{
    color: #236ce6!important;
    text-shadow: 0 0 3vw #F40A35!important;
}}
.haiqu_pink {{
    color: #f7cfc6!important;
}}
.haiqu_light {{
    color: #ffeabf!important;
}}
.haiqu_neutral {{
    color: #d9d9d9!important;
}}
.haiqu_light2 {{
    color: #cde9f7!important;
}}
</style>

<div id="{}" class="haiqu_widget {} vectors_bg">
    <p class="haiqu_widget_title">
        {} <span>{}</span>
    </p>
    <p class="haiqu_widget_title_bottom"></p>
    {}
</div>
"""  # noqa

JS_CANVES_RENDERER = """
(async () => {{
    // haiqu-widget-timestamp
    const el = document.querySelector('#{}');
    // output element
    const output = document.querySelector('.log-input-{}').querySelector("input");

    if (!window.html2canvas) {{
        await new Promise((resolve, reject) => {{
            const s = document.createElement('script');
            s.src = "https://cdn.jsdelivr.net/npm/html2canvas-pro@1.6.4/dist/html2canvas-pro.min.js";
            s.onload = resolve;
            s.onerror = reject;
            document.head.appendChild(s);
        }});
    }}

    console.log("Rendering...");

    const canvas = await window.html2canvas(el, {{
        scale: 1
    }});
    const dataUrl = canvas.toDataURL("image/png");

    // set data URL to the output element, fire change event for Jupyter to detect the change
    output.value = dataUrl;
    const event = new Event('change', {{ bubbles: true }});
    output.dispatchEvent(event);
}})();
"""  # noqa


LOG_WIDGET_HELP_TEXT = """No worries, here are examples to help you get started with logging in Haiqu Jupyter Lab:

circuit = QuantumCircuit(2)
haiqu.log(circuit)

haiqu.log(12.34, name="Some value")
haiqu.log("Hello quantum world!", name="Some textual value")
haiqu.log([1, 2, 3], name="Experiment parameters")

import matplotlib.pyplot as plt
plt.plot(...)
haiqu.log(plt, name="Matplotlib plot")

fig, ax = plt.subplots()
haiqu.log(fig, name="Matplotlib figure")

from haiqu.sdk.wiz.drawer import Drawer
drawer = Drawer()
drawer.plot(...)
haiqu.log(drawer, name="Cool drawer plot")

haiqu.log({"chart": plt})
haiqu.log({"examples": ["one", "two", "three"]})
haiqu.log({"some_value": 12.34, "some_text": "Quantum!", "parameters": [1, 2, 3]})

You can see logged data on Dashboard: https://dashboard.haiqu.ai"""


def render_template(
    title: str,
    html_str: str,
    widget_extra_class: str = "",
    height: str = "150px",
    widget_id: str = "",
) -> str:
    """
    Renders widget from the template.

    Args:
        title (str): The title of the widget.
        html_str (str): The rendered HTML body - plot or table with data.
        widget_extra_class (str): The extra CSS class to include into widget body.
        height (str): The height.
        widget_id (str): The widget ID. Optional

    Returns:
        str: The rendered template for Widget.
    """
    if not widget_id:
        widget_id = str(uuid.uuid4()).replace("-", "")

    return TEMPLATE_WIDGET.format(height, widget_id, widget_extra_class, HAIQU_LOGO, title, html_str)


def draw_core_analytics(
    circuit: schemas.CircuitModel,
    help: bool = False,
    tiles_layout: bool = False,
):
    """
    Render core analytics
    """
    html_str = ""
    if tiles_layout:
        html_str += plot_radar(circuit, help=help, tiles_layout=tiles_layout)
        html_str += plot_gate_diversity(circuit, help=help, tiles_layout=tiles_layout)
        html_str += metrics_as_table(circuit, help=help, tiles_layout=tiles_layout)
        html_str += plot_gate_diversity(circuit, help=help, tiles_layout=tiles_layout, basis_gates=True)
        html_str += plot_liveness_per_qubit(circuit, help=help, tiles_layout=tiles_layout)
        html_str += plot_correlation_matrix(circuit, help=help, tiles_layout=tiles_layout)
    else:
        html_str += metrics_as_table(circuit, help=help, tiles_layout=tiles_layout)
        html_str += plot_radar(circuit, help=help, tiles_layout=tiles_layout)
        html_str += plot_gate_diversity(circuit, help=help, tiles_layout=tiles_layout)
        html_str += plot_gate_diversity(circuit, help=help, tiles_layout=tiles_layout, basis_gates=True)
        html_str += plot_liveness_per_qubit(circuit, help=help, tiles_layout=tiles_layout)
        html_str += plot_correlation_matrix(circuit, help=help, tiles_layout=tiles_layout)

    return display(HTML(html_str))


def metrics_as_table(
    circuit: schemas.CircuitModel,
    help: bool = False,
    tiles_layout: bool = False,
    core_only: bool = False,
    advanced_only: bool = False,
    widget_id: str = "",
) -> str:
    """
    Render circuit core metrics in the table format.
    """
    metrics = circuit.analytics

    core_data = [
        ["CIRCUIT", circuit.name, ""],
        ["TOTAL QUBITS", metrics.qubits, "Total number of qubits in the circuit."],
        ["ACTIVE QUBITS", metrics.num_qubits_active, "Number of active (non-idle) qubits in the circuit."],
        ["NUM PARAMETERS", metrics.num_parameters, "Number of parameters in the circuit."],
        ["DEPTH", metrics.depth, "The number of sequential operation layers (original circuit)."],
        ["2-Q GATES DEPTH", metrics.depth_2q, "The number of sequential layers of 2-qubit gates."],
        ["1-Q GATES", metrics.gates_1q, "Count of 1-qubit gates applied."],
        ["2-Q GATES", metrics.gates_2q, "Count of 2-qubit entangling gates."],
        ["OTHER GATES", metrics.other_gates, "Count of gates that are neither 1-qubit nor 2-qubit gates."],
        ["GATES TOTAL", metrics.gates_total, "Total number of gates in the circuit."],
        [
            "OTHER OPERATIONS",
            metrics.other_ops,
            "Count of non-gate operations such as measurements, barriers, resets, or delays.",
        ],
        [
            "INSTRUCTIONS TOTAL",
            metrics.instructions_total,
            "Total number of circuit instructions, including gates and non-gate operations.",
        ],
    ]
    advanced_data = [
        [
            "PROGRAM COMMUNICATION",
            safe_format_float(metrics.program_communication),
            "Evaluates qubit connectivity by computing the sum of interaction degrees in the circuit's interaction graph.",
        ],
        [
            "CRITICAL DEPTH",
            safe_format_float(metrics.critical_depth),
            """Determines how deep the two-qubit interactions are relative to their total count,
            assessing the circuit's parallel execution potential.""",
        ],
        [
            "ENTANGLING GATES RATIO",
            safe_format_float(metrics.entanglement_ratio),
            "Calculates the fraction of two-qubit gates in the circuit, indicating how much entanglement is present.",
        ],
        [
            "PARALLELISM",
            safe_format_float(metrics.parallelism),
            """Measures the circuit's ability to execute operations simultaneously
            by comparing total gate count to circuit depth.""",
        ],
        [
            "LIVENESS",
            safe_format_float(metrics.liveness),
            """Computes how often qubits are active during execution relative to total circuit depth,
            indicating qubit utilization.""",
        ],
        [
            "KL DIVERGENCE",
            safe_format_float(metrics.kl_divergence),
            """Kullback-Leibler divergence is used to measure how closely
            the final state distribution of a circuit approximates a uniform distribution.""",
        ],
    ]

    if core_only:
        data = core_data
    elif advanced_only:
        data = advanced_data
    else:
        data = core_data + advanced_data

    html_str = """<table><tbody>"""
    for row in data:
        if help:
            html_str += f"""<tr>
                <td style="width: 250px!important" nowrap>{row[0]}</td>
                <td style="text-align:right!important;width: 100px!important" nowrap>{row[1]}</td>
                <td>{row[2]}</td>
            </tr>"""
        else:
            html_str += f"""<tr>
                <td nowrap>{row[0]}</td>
                <td style="text-align:right!important" nowrap>{row[1]}</td>
            </tr>"""
    html_str += """</tbody></table>"""

    widget_extra_class = ""
    if tiles_layout:
        widget_extra_class = "haiqu_widget_tile_sm"

    return render_template(
        title="INPUT CIRCUIT DETAILS",
        html_str=html_str,
        widget_extra_class=widget_extra_class,
        widget_id=widget_id,
    )


def compare_metrics_as_table(
    circuits: list[schemas.CircuitModel],
    help: bool = False,
    tiles_layout: bool = False,
    widget_id: str = "",
):
    """
    Render comparison of circuit core metrics in the table format.
    """
    metrics = [circuit.analytics for circuit in circuits]

    data = [
        ["CIRCUIT"] + [circuit.name for circuit in circuits] + [""],
        ["TOTAL QUBITS"] + [m.qubits for m in metrics] + ["Total number of qubits in the circuit."],
        ["ACTIVE QUBITS"] + [m.num_qubits_active for m in metrics] + ["Number of active (non-idle) qubits in the circuit."],
        ["NUM PARAMETERS"] + [m.num_parameters for m in metrics] + ["Number of parameters in the circuit."],
        ["DEPTH"] + [m.depth for m in metrics] + ["The number of sequential operation layers (original circuit)."],
        ["2-Q GATES DEPTH"] + [m.depth_2q for m in metrics] + ["The number of sequential layers of 2-qubit gates."],
        ["1-Q GATES"] + [m.gates_1q for m in metrics] + ["Count of 1-qubit gates applied."],
        ["2-Q GATES"] + [m.gates_2q for m in metrics] + ["Count of 2-qubit entangling gates."],
        ["OTHER GATES"] + [m.other_gates for m in metrics] + ["Count of gates that are neither 1-qubit nor 2-qubit gates."],
        ["GATES TOTAL"] + [m.gates_total for m in metrics] + ["Total number of gates in the circuit."],
        ["OTHER OPERATIONS"]
        + [m.other_ops for m in metrics]
        + ["Count of non-gate operations such as measurements, barriers, resets, or delays."],
        ["INSTRUCTIONS TOTAL"]
        + [m.instructions_total for m in metrics]
        + ["Total number of circuit instructions, including gates and non-gate operations."],
    ]

    html_str = """<table><tbody>"""
    for row in data:
        # Double bold the smaller value (skip CIRCUIT name row)
        values = row[1:-1]
        if all(isinstance(v, (int, float)) for v in values):
            min_value = min(values)
        else:
            min_value = None
        values_html = []
        for v in values:
            if v == min_value:
                values_html.append(f"<b><b>{v}</b></b>")
            else:
                values_html.append(f"{v}")

        if help:
            html_str += "<tr>"
            html_str += f"""<td style="width: 250px!important" nowrap>{row[0]}</td>"""
            for val_html in values_html:
                html_str += f"""<td style="text-align:right!important;width: 100px!important" nowrap>{val_html}</td>"""
            html_str += f"""<td>{row[-1]}</td>"""
            html_str += "</tr>"
        else:
            html_str += "<tr>"
            html_str += f"""<td nowrap>{row[0]}</td>"""
            for val_html in values_html:
                html_str += f"""<td style="text-align:right!important" nowrap>{val_html}</td>"""
            html_str += "</tr>"
    html_str += """</tbody></table>"""
    widget_extra_class = ""
    if tiles_layout:
        widget_extra_class = "haiqu_widget_tile_sm"

    return render_template(
        title="CIRCUIT COMPARISON",
        html_str=html_str,
        widget_extra_class=widget_extra_class,
        widget_id=widget_id,
    )


def plot_radar(
    circuit: schemas.CircuitModel,
    help: bool = False,
    tiles_layout: bool = False,
):
    """Radar plot for the circuit properties"""
    layout = go.Layout(
        paper_bgcolor="rgba(25,25,25,1)",
        plot_bgcolor="rgba(25,25,25,1)",
        font_color="rgba(255,255,255,1)",
        legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.3),
    )

    fig = go.Figure(layout=layout)
    benchmarks = circuit.benchmarks

    for b in benchmarks:
        fig.add_trace(
            go.Scatterpolar(
                r=[b.program_communication, b.critical_depth, b.entanglement_ratio, b.parallelism, b.liveness],
                theta=["Program Communication", "Critical Depth", "Entangling Gates Ratio", "Parallelism", "Liveness"],
                fill="toself",
                name=b.name,
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1.2]),
            angularaxis=dict(showgrid=False, showline=False),
            bgcolor="rgba(25, 25, 25, 1)",
        ),
        showlegend=True,
        height=550,
    )
    html_str = fig.to_html()

    if help:
        # TODO: Add real "Read more..." link.
        html_str += """<table><tbody>
<tr>
    <td>Program Communication</td>
    <td>Evaluates qubit connectivity by computing the sum of interaction degrees
    in the circuit's interaction graph.</td>
</tr>
<tr>
    <td>Critical Depth</td>
    <td>Determines how deep the two-qubit interactions are relative to their total count,
    assessing the circuit's parallel execution potential.</td>
</tr>
<tr>
    <td>Entangling Gates Ratio</td>
    <td>Calculates the fraction of two-qubit gates in the circuit,
    indicating how much entanglement is present.</td>
</tr>
<tr>
    <td>Parallelism</td>
    <td>Measures the circuit's ability to execute operations simultaneously
    by comparing total gate count to circuit depth.</td>
</tr>
<tr>
    <td>Liveness</td>
    <td>Computes how often qubits are active during execution relative to total circuit depth,
    indicating qubit utilization.</td>
</tr>
<tr>
    <td>Reference</td>
    <td><a target=_blank href="https://docs.haiqu.ai">Read more...</a></td>
</tr></tbody></table>"""

    widget_extra_class = ""
    if tiles_layout:
        widget_extra_class = "haiqu_widget_tile"

    return render_template("INPUT CIRCUIT PROPERTIES", html_str, widget_extra_class)


def plot_gate_diversity(
    circuit: schemas.CircuitModel,
    help: bool = False,
    tiles_layout: bool = False,
    basis_gates: bool = False,
):
    """
    Donut plot for gate diversity.

    Args:
        circuit (CircuitModel): The circuit metadata object.
        basis_gates (bool): Wherever to plot gate diversity with normalized circuit or original one.
    """
    if basis_gates:
        title = "BASIS GATES DIVERSITY"
        data = circuit.analytics.gate_diversity_basis_gates
    else:
        title = "GATES DIVERSITY"
        data = circuit.analytics.gate_diversity

    # in case analytics computation returned error
    if isinstance(data, str):
        return render_template(title, f"""<pre>{data}</pre>""")

    labels = list(data.keys())
    values = list(data.values())

    layout = go.Layout(
        paper_bgcolor="rgba(25,25,25,1)",
        plot_bgcolor="rgba(25,25,25,1)",
        font_color="rgba(255,255,255,1)",
    )

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.9)], layout=layout)
    fig.update_traces(marker=dict(line=dict(color="#ffffff", width=1)))
    html_str = fig.to_html()

    if help:
        # TODO: Add real "Read more..." link.
        html_str += """<table><tbody>
<tr>
    <td>Definition</td>
    <td>Computes the relative frequency of specific gate types (e.g., rx, ry, rz, cx),
    providing insight into circuit structure.</td>
</tr>
<tr>
    <td>Importance</td>
    <td>Helps assess how diverse the circuit operations are.</td>
</tr>
<tr>
    <td>Usage</td>
    <td>Useful for hardware compatibility and circuit simplification.</td>
</tr>
<tr>
    <td>Reference</td>
    <td><a target=_blank href="https://docs.haiqu.ai">Read more...</a></td>
</tr></tbody></table>"""

    widget_extra_class = ""
    if tiles_layout:
        widget_extra_class = "haiqu_widget_tile"

    return render_template(title, html_str, widget_extra_class)


def plot_liveness_per_qubit(
    circuit: schemas.CircuitModel,
    help: bool = False,
    tiles_layout: bool = False,
):
    """Bar chart for liveness per qubit."""
    labels = list(range(len(circuit.analytics.liveness_per_qubit)))
    values = circuit.analytics.liveness_per_qubit
    layout = go.Layout(
        paper_bgcolor="rgba(25,25,25,1)",
        plot_bgcolor="rgba(25,25,25,1)",
        font_color="rgba(255,255,255,1)",
    )

    fig = go.Figure(layout=layout)
    fig.add_trace(go.Bar(y=labels, x=values, orientation="h"))
    html_str = fig.to_html()

    if help:
        # TODO: Add real "Read more..." link.
        html_str += """<table><tbody>
<tr>
    <td>Definition</td>
    <td>Determines individual qubit activity across the circuit depth,
    providing a more detailed view of resource usage.</td>
</tr>
<tr>
    <td>Importance</td>
    <td>Prevents qubit decoherence by keeping them active.</td>
</tr>
<tr>
    <td>Usage</td>
    <td>Helps optimize circuits for coherence time.</td>
</tr>
<tr>
    <td>Reference</td>
    <td><a target=_blank href="https://docs.haiqu.ai">Read more...</a></td>
</tr></tbody></table>"""

    widget_extra_class = ""
    if tiles_layout:
        widget_extra_class = "haiqu_widget_tile"

    return render_template("LIVENESS PER QUBIT", html_str, widget_extra_class)


def plot_correlation_matrix(
    circuit: schemas.CircuitModel,
    help: bool = False,
    tiles_layout: bool = False,
):
    """Matrix/heatmap chart for correlation matrix."""
    num_qubits = circuit.analytics.qubits
    tickvals = list(range(num_qubits))
    dtick = 1
    height = 600

    if num_qubits > 50:
        tickvals = None
        dtick = 10
        height = 800
    elif num_qubits > 20:
        tickvals = None
        dtick = 5
        height = 700

    layout = go.Layout(
        paper_bgcolor="rgba(25,25,25,1)",
        plot_bgcolor="rgba(25,25,25,1)",
        font_color="rgba(255,255,255,1)",
        xaxis=dict(title="", tickvals=tickvals, side="top", dtick=dtick, showgrid=False, constrain="domain"),
        yaxis=dict(
            title="", tickvals=tickvals, autorange="reversed", dtick=dtick, scaleanchor="x", showgrid=False, constrain="domain"
        ),
        height=height,
    )

    c = circuit.analytics.correlation_matrix
    plot_data = [[c.get(f"{x},{y}", 0.0) for x in range(num_qubits)] for y in range(num_qubits)]

    fig = go.Figure(data=go.Heatmap(z=plot_data, colorscale=[[0, "#000000"], [1, "#236CE6"]], zmin=0), layout=layout)
    html_str = fig.to_html()

    if help:
        # TODO: Add real "Read more..." link.
        html_str += """<table><tbody>
<tr>
    <td>Definition</td>
    <td>A heatmap showing interactions between qubits.</td>
</tr>
<tr>
    <td>Importance</td>
    <td>Visualizes entanglement and qubit connectivity.</td>
</tr>
<tr>
    <td>Usage</td>
    <td>Helps optimize qubit placement on hardware.</td>
</tr>
<tr>
    <td>Reference</td>
    <td><a target=_blank href="https://docs.haiqu.ai">Read more...</a></td>
</tr></tbody></table>"""

    widget_extra_class = ""
    if tiles_layout:
        widget_extra_class = "haiqu_widget_tile"

    return render_template("CORRELATION MATRIX BETWEEN QUBITS", html_str, widget_extra_class)


ALLOWED_X_EVOLUTION_METRICS = ["depth", "gates_1q", "gates_2q", "gates_total"]


def plot_circuit_evolution(
    circuit: schemas.CircuitModel,
    metric: str = "depth",
    help: bool = False,
    tiles_layout: bool = False,
):
    """Chart for circuit evolution."""
    evolution = circuit.evolution.metrics

    if metric not in ALLOWED_X_EVOLUTION_METRICS:
        raise ValueError(f"`metric` must be one of {ALLOWED_X_EVOLUTION_METRICS}")

    widget_extra_class = ""
    if tiles_layout:
        widget_extra_class = "haiqu_widget_tile"

    # in case analytics computation returned error
    if isinstance(evolution, str):
        return display(HTML(render_template("CIRCUIT EVOLUTION", f"""<pre>{evolution}</pre>""", widget_extra_class)))

    metric_labels = {  # list of labels for a plot; acts also as a list of metrics we plot
        "entanglement_ratio": "Entangling Gates Ratio",
        "program_communication": "Program Communication",
        "critical_depth": "Critical Depth",
        "parallelism": "Parallelism",
        "liveness": "Liveness",
    }

    evolution_values = defaultdict(dict)

    max_slice_metric_value = None
    for point in evolution:
        slice_metric_value = point[metric]
        for label in metric_labels:
            if label in point:
                evolution_values[label][slice_metric_value] = point[label]
        max_slice_metric_value = slice_metric_value

    layout = go.Layout(
        paper_bgcolor="rgba(25,25,25,1)",
        plot_bgcolor="rgba(25,25,25,1)",
        font_color="rgba(255,255,255,1)",
        legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.3),
    )

    fig = go.Figure(layout=layout)
    # there are 10 steps in evolution computation. Metrics at 0 are missing, but it is better to have 0 on the axis
    fig.update_layout(xaxis_title=metric, xaxis_range=[0, max_slice_metric_value * 1.05])

    for label in metric_labels:
        if len(evolution_values[label]) > 0:
            fig.add_trace(
                go.Scatter(
                    x=list(evolution_values[label].keys()),
                    y=list(evolution_values[label].values()),
                    mode="lines+markers",
                    line_shape="spline",
                    name=metric_labels[label],
                )
            )

    html_str = fig.to_html()

    if help:
        # TODO: Add real "Read more..." link.
        html_str += """<table><tbody>
<tr>
    <td>Program Communication</td>
    <td>Evaluates qubit connectivity by computing the sum of interaction degrees in the circuit's interaction graph.</td>
</tr>
<tr>
    <td>Critical Depth</td>
    <td>Determines how deep the two-qubit interactions are relative to their total count,
    assessing the circuit's parallel execution potential.</td>
</tr>
<tr>
    <td>Entangling Gates Ratio</td>
    <td>Calculates the fraction of two-qubit gates in the circuit, indicating how much entanglement is present.</td>
</tr>
<tr>
    <td>Parallelism</td>
    <td>Measures the circuit's ability to execute operations simultaneously by comparing total gate count to circuit depth.</td>
</tr>
<tr>
    <td>Liveness</td>
    <td>Computes how often qubits are active during execution relative to total circuit depth, indicating qubit utilization.</td>
</tr>
<tr>
    <td>Reference</td>
    <td><a target=_blank href="https://docs.haiqu.ai">Read more...</a></td>
</tr></tbody></table>"""

    return display(HTML(render_template("CIRCUIT EVOLUTION", html_str, widget_extra_class)))


def list_experiments(items: list[schemas.ExperimentModel]):
    """
    Render experiments widget.
    """
    html_str = """<table width="100%"><tr>
        <th style="font-weight:bold;text-align:left!important">Name</th>
        <th style="font-weight:bold;text-align:left!important">ID</th>
        <th style="font-weight:bold;text-align:left!important">Creation date (UTC)</th>
        <th style="font-weight:bold;text-align:left!important">Last action date (UTC)</th>
        <th style="font-weight:bold;text-align:left!important">Circuits</th>
        <th style="font-weight:bold;text-align:left!important">Jobs</th>
    </tr>"""
    for item in items:
        html_str += f"""<tr>
            <td style="width: 100px;text-align:left;!important" nowrap>{item.name}</td>
            <td style="width: 100px;text-align:left!important" nowrap>{item.id}</td>
            <td style="width: 100px;text-align:left!important" nowrap>{item.creation_date.strftime(DATE_TIME_FORMAT)}</td>
            <td style="width: 100px;text-align:left!important" nowrap>
                {item.last_action_date.strftime(DATE_TIME_FORMAT) if item.last_action_date is not None else "-"}
            </td>
            <td style="width: 100px;text-align:left!important" nowrap>{item.circuits_count}</td>
            <td style="width: 100px;text-align:left!important" nowrap>{item.jobs_count}</td>
        </tr>"""
    html_str += """</table>"""
    return display(HTML(render_template("EXPERIMENTS", html_str, height="150px")))


def list_circuits(items: list[schemas.CircuitModel], title: str = "CIRCUITS"):
    """
    Render circuits widget.
    """
    html_str = """<table width="100%"><tr>
        <th style="font-weight:bold;text-align:left!important">Name</th>
        <th style="font-weight:bold;text-align:left!important">ID</th>
        <th style="font-weight:bold;text-align:left!important">Creation date (UTC)</th>
        <th style="font-weight:bold;text-align:left!important">Description</th>
        <th style="font-weight:bold;text-align:left!important">Number of runs</th>
    </tr>"""
    for item in items:
        jobs_count = len(item.job_ids) if item.job_ids is not None else 0
        html_str += f"""<tr>
            <td style="width: 100px;text-align:left;!important" nowrap>{item.name}</td>
            <td style="width: 100px;text-align:left!important" nowrap>{item.id}</td>
            <td style="width: 100px;text-align:left!important" nowrap>{item.creation_date.strftime(DATE_TIME_FORMAT)}</td>
            <td style="width: 100px;text-align:left!important" nowrap>{item.description or ""}</td>
            <td style="width: 100px;text-align:left!important" nowrap>{jobs_count}</td>
        </tr>"""
    html_str += """</table>"""
    return display(HTML(render_template(title, html_str, height="150px")))


def list_transpiled_circuits(items: list[schemas.CircuitModel], title: str = "TRANSPILED CIRCUITS"):
    """
    Render transpiled circuits widget.
    """
    html_str = """<table width="100%"><tr>
        <th style="font-weight:bold;text-align:left!important">Name</th>
        <th style="font-weight:bold;text-align:left!important">ID</th>
        <th style="font-weight:bold;text-align:left!important">Creation date (UTC)</th>
        <th style="font-weight:bold;text-align:left!important">Transpilation options</th>
        <th style="font-weight:bold;text-align:left!important">Target device</th>
        <th style="font-weight:bold;text-align:left!important">Number of runs</th>
    </tr>"""
    for item in items:
        jobs_count = len(item.job_ids) if item.job_ids is not None else 0
        transpilation_options = item.transpilation_options or {}
        transpilation_str = ", ".join([f"{k}: {v}" for k, v in transpilation_options.items()])
        html_str += f"""<tr>
            <td style="width: 100px;text-align:left;!important" nowrap>{item.name}</td>
            <td style="width: 100px;text-align:left!important" nowrap>{item.id}</td>
            <td style="width: 100px;text-align:left!important" nowrap>{item.creation_date.strftime(DATE_TIME_FORMAT)}</td>
            <td style="width: 100px;text-align:left!important" nowrap>{transpilation_str}</td>
            <td style="width: 100px;text-align:left!important" nowrap>{item.transpilation_target or ""}</td>
            <td style="width: 100px;text-align:left!important" nowrap>{jobs_count}</td>
        </tr>"""
    html_str += """</table>"""
    return display(HTML(render_template("TRANSPILED CIRCUITS", html_str, height="150px")))


def list_artifacts(items: list[schemas.ArtifactModel]):
    """
    Render artifacts widget, ordered by name.
    """
    html_str = """<table width="100%"><tr>
        <th style="font-weight:bold;text-align:left!important">Name</th>
        <th style="font-weight:bold;text-align:left!important">Type</th>
        <th style="font-weight:bold;text-align:left!important">Data</th>
        <th style="font-weight:bold;text-align:left!important">Creation date (UTC)</th>
        <th style="font-weight:bold;text-align:left!important">Last updated (UTC)</th>
    </tr>"""
    for item in sorted(items, key=lambda item: item.name):
        if item.artifact_data is None:
            artifact_data = "-"
        else:
            if item.artifact_type == "timeseries":
                artifact_data = str(list(item.artifact_data.values()))
            else:
                if not item.artifact_data:
                    artifact_data = "-"
                else:
                    artifact_data = str(list(item.artifact_data.values())[-1])

        if len(artifact_data) > 30:
            artifact_data = artifact_data[:30] + "..."
        html_str += f"""<tr>
            <td style="width: 100px;text-align:left;!important" nowrap>{item.name}</td>
            <td style="width: 100px;text-align:left!important" nowrap>{item.artifact_type}</td>
            <td style="width: 100px;text-align:left!important" nowrap>{artifact_data}</td>
            <td style="width: 100px;text-align:left!important" nowrap>
                {item.creation_date.strftime(DATE_TIME_FORMAT) if item.creation_date is not None else "-"}
            </td>
            <td style="width: 100px;text-align:left!important" nowrap>
                {item.last_updated.strftime(DATE_TIME_FORMAT) if item.last_updated is not None else "-"}
            </td>
        </tr>"""
    html_str += """</table>"""
    return display(HTML(render_template("ARTIFACTS", html_str, height="150px")))


def list_jobs(items: list[schemas.BaseJobModel]):
    """
    Render jobs widget.
    """

    def _format_job_parameters(job) -> str:
        """Format job parameters for display in the table. Same logic as in Dashboard."""
        parameters = job.parameters

        if not parameters:
            return None
        if isinstance(parameters, dict):
            parameters_str = ", ".join([f"{k}={v}" for k, v in parameters.items()])
        elif isinstance(parameters, list):
            parameters_str = ", ".join([str(p) for p in parameters])

        if len(parameters_str) > 45:
            parameters_str = parameters_str[:45] + "..."
        return parameters_str

    html_str = """<table width="100%"><tr>
        <th style="font-weight:bold;text-align:left!important">Job</th>
        <th style="font-weight:bold;text-align:left!important">Status</th>
        <th style="font-weight:bold;text-align:left!important">Type</th>
        <th style="font-weight:bold;text-align:left!important">Backend</th>
        <th style="font-weight:bold;text-align:left!important">Created (UTC)</th>
        <th style="font-weight:bold;text-align:left!important">QPU Quality</th>
        <th style="font-weight:bold;text-align:left!important">CPU Time</th>
        <th style="font-weight:bold;text-align:left!important">QPU Cost</th>
    </tr>"""
    for item in items:
        name_column = f"""<span style="font-size:.625rem">{item.id}</span>
            <br />
            {item.name or item.experiment.name}
            <br />
            <span style="font-size:.625rem">{item.description or _format_job_parameters(item) or "No description"}</span>
        """

        # Got from Dashboard
        qpu_cost = (
            item.info.get("qpu_cost", {}).get("converted", {}).get("amount")
            if item.info is not None and item.info.get("qpu_cost") is not None and item.status == schemas.JobStatus.DONE
            else None
        )

        html_str += f"""<tr>
            <td style="min-width:300px;max-width:300px;text-align:left;!important;word-break: break-all;" nowrap>
                {name_column}
            </td>
            <td style="max-width:300px;text-align:left;!important;word-break: break-all;" nowrap>{item.status.value}</td>
            <td style="max-width:300px;text-align:left;!important;word-break: break-all;" nowrap>{item.job_type.value}</td>
            <td style="max-width:300px;text-align:left;!important;word-break: break-all;" nowrap>{item.device_id}</td>
            <td style="max-width:300px;text-align:left!important;word-break: break-all;" nowrap>
                {item.creation_date.strftime(DATE_TIME_FORMAT_JOBS)}</td>
            <td style="max-width:300px;text-align:left!important;word-break: break-all;" nowrap>
                { "{:.4f}".format(item.quality) if item.quality is not None else "-" }
            </td>
            <td style="max-width:300px;text-align:left!important;word-break: break-all;" nowrap>
                { "{:.2f} s".format(item.time) if item.time is not None else "-" }
            </td>
            <td style="max-width:300px;text-align:left!important;word-break: break-all;" nowrap>
                { "${:.2f}".format(qpu_cost) if qpu_cost is not None else "-" }
            </td>
        </tr>"""
    return display(HTML(render_template("JOBS", html_str, height="150px")))


def list_devices(items: list[schemas.DeviceModel]):
    """
    Render quantum devices/backend widget.
    """
    html_str = """<table width="100%"><tr>
        <th style="font-weight:bold;text-align:left!important">Vendor</th>
        <th style="font-weight:bold;text-align:left!important">Quantum System</th>
        <th style="font-weight:bold;text-align:left!important">Qubits</th>
        <th style="font-weight:bold;text-align:left!important">Status</th>
        <th style="font-weight:bold;text-align:left!important">Queue Length</th>
    </tr>"""
    # sort items by vendor name and then by qubits count descending
    items = sorted(items, key=lambda x: (x.vendor, -x.qubits))
    for item in items:
        html_str += f"""<tr>
            <td style="width: 30px!important" nowrap>{item.vendor}</td>
            <td style="width: 100px;text-align:left!important" nowrap>{item.id}</td>
            <td style="width: 100px;text-align:left!important" nowrap>{item.qubits}</td>
            <td style="width: 100px;text-align:left!important" nowrap>{item.status}</td>
            <td style="width: 100px;text-align:left!important" nowrap>{item.pending_jobs or str(0)}</td>
        </tr>"""
    html_str += """</table>"""
    return display(HTML(render_template("QUANTUM DEVICES", html_str, height="150px")))


def list_simulators(items: list[schemas.DeviceModel]):
    """
    Render quantum simulators/backend widget.
    """
    html_str = """<table width="100%"><tr>
        <th style="font-weight:bold;text-align:left!important">Vendor</th>
        <th style="font-weight:bold;text-align:left!important">Quantum System</th>
        <th style="font-weight:bold;text-align:left!important">Qubits</th>
        <th style="font-weight:bold;text-align:left!important">Status</th>
    </tr>"""
    # sort items by vendor name and then by qubits count descending
    items = sorted(items, key=lambda x: (x.vendor, -x.qubits))
    for item in items:
        html_str += f"""<tr>
            <td style="width: 30px!important" nowrap>{item.vendor}</td>
            <td style="width: 100px;text-align:left!important" nowrap>{item.id}</td>
            <td style="width: 100px;text-align:left!important" nowrap>{item.qubits}</td>
            <td style="width: 100px;text-align:left!important" nowrap>{item.status}</td>
        </tr>"""
    html_str += """</table>"""
    return display(HTML(render_template("QUANTUM SIMULATORS", html_str, height="150px")))


def draw_neon_circuit(circuit, style: str = ""):
    """
    Render the qiskit.QuantumCircuit. Neon, Japan 80s style, glowing.

    Args:
        circuit (QuantumCircuit): The quantum circuit to draw.
        style (str): The extra CSS class to use.

    Returns:
        None
    """
    safe_str = re.sub(r"<.*?>", "", str(circuit.draw(output="text", fold=-1)))
    html_str = f"""<pre class="neonQASM {style}">{safe_str}</pre>"""
    display(HTML(render_template(f"QUANTUM CIRCUIT: {circuit.name}", html_str)))


def benchmarks_as_table(benchmarks: Iterable, format: str) -> Union[Iterable, str]:
    """Display circuit key metrics alongside public benchmarks.

    Various plain-text table formats (`format`) are supported:
    'plain', 'simple', 'grid', 'pipe', 'orgtbl', 'rst', 'mediawiki',
    'latex', 'latex_raw', 'latex_booktabs', 'latex_longtable' and tsv.

    Args:
        benchmarks (Iterable): Circuit and reference circuits/benchmarks.
        format (str): The output format. Default is raw result.

    Returns:
        Union[Iterable, str]: The benchmarks list or formatted string.
    """

    return tabulate(
        [[field[1] for field in x] for x in benchmarks],
        headers=[
            "Program communication",
            "Critical depth",
            "Entanglement ratio",
            "Parallelism",
            "Liveness",
        ],
        tablefmt=format,
    )


def draw_data_loading_estimates(
    data: schemas.DataLoadingEstimatesModel,
    help: bool = False,
    format: str = "html",
    timefmt: str = "{:.2f} seconds",
    moneyfmt: str = "{:.2f} credits",
    tiles_layout: bool = False,
) -> str:
    """
    Render the widget with distribution data loading estimates.
    """

    html_str = tabulate(
        [
            ["COST", moneyfmt.format(data.estimated_cost)],
            ["TIME", timefmt.format(data.estimated_time)],
        ],
        tablefmt=format,
    )

    widget_extra_class = ""
    if tiles_layout:
        widget_extra_class = "haiqu_widget_tile_sm"

    return display(HTML(render_template("ESTIMATES FOR DISTRIBUTION DATA LOADING", html_str, widget_extra_class)))


def compression_estimates(
    data: schemas.StateCompressionEstimatesModel,
    help: bool = False,
    format: str = "html",
    timefmt: str = "{:.2f} seconds",
    moneyfmt: str = "{:.2f} credits",
    tiles_layout: bool = False,
) -> str:
    """
    Render the widget with compression estimates.
    """

    html_str = tabulate(
        [
            ["COST", moneyfmt.format(data.estimated_cost)],
            ["TIME", timefmt.format(data.estimated_time)],
        ],
        tablefmt=format,
    )

    widget_extra_class = ""
    if tiles_layout:
        widget_extra_class = "haiqu_widget_tile_sm"

    return display(HTML(render_template("ESTIMATES FOR STATE COMPRESSION", html_str, widget_extra_class)))


def job_progress_widget(title: str, data: str):
    """
    Render the widget with the job logs stream.
    """

    data = data or "Waiting for data..."

    html_str = f"""
<pre>
{data}
</pre>
"""
    display(HTML(render_template(title, html_str)), clear=True)


def safe_format_float(val, format="%.6f"):
    """Format metric if float with given precision."""
    if isinstance(val, float):
        return format % val
    return val


def generate_widget_id():
    """Generate a unique widget ID based on the current timestamp."""
    return f"haiqu-widget-{datetime.now().strftime('%Y%m%d%H%M%S')}"


def loggable_widget(
    circuit: schemas.CircuitModel,
    artifact_type: str,
    html_str: str,
    widget_id: str = "",
    multiple_circuits: bool = False,
):
    """Wrap widget HTML to be loggable."""

    button = widgets.Button(
        icon="cloud-upload",
        description=" Log to Experiment",
        layout=widgets.Layout(width="97%"),
    )

    text_widget = widgets.Text(layout=widgets.Layout(visibility="hidden"))
    text_widget._dom_classes = text_widget._dom_classes + (f"log-input-{widget_id}",)
    dashboard_link = widgets.Output(layout=widgets.Layout(padding="0px", margin="0px"))

    def data_ready(data):
        "Called by Jupyter when data is ready to be logged."
        value = data["new"]

        if not value:
            button.description = " Something went wrong, please try again later."
            return

        timestamp = datetime.now().strftime(DATE_TIME_FORMAT)
        if multiple_circuits:
            artifact_name = f"{artifact_type} / {timestamp}"
        else:
            artifact_name = f"{artifact_type}: {circuit.name} / {timestamp}"

        circuit._client.submit_experiment_metrics(
            experiment_id=circuit.experiment_id,
            data=schemas.SubmitMetricsModel(metrics={artifact_name: value}),
        )
        button.description = " Done. Artifact is available on the Dashboard."
        dashboard_link.clear_output()
        dashboard_link.append_stdout("Click here to view logged artifact on dashboard:\n")
        dashboard_link.append_stdout(f"https://dashboard.haiqu.ai/experiment/{circuit.experiment_id}/")

    text_widget.observe(data_ready, names="value")

    js = Javascript(JS_CANVES_RENDERER.format(widget_id, widget_id))

    output = widgets.Output(layout=widgets.Layout(padding="0px", margin="0px"))

    @output.capture(clear_output=True, wait=True)
    def on_log_button_clicked(btn):
        btn.description = " Logging..."
        btn.disabled = True
        sleep(0.1)
        display(js)

    button.on_click(on_log_button_clicked)

    display(HTML(html_str), button, dashboard_link, text_widget, output)


def log_error_widget(data: str):
    """
    Render the widget with the log error and help.
    """

    title = "Oops!"
    html_str = f"""
<pre>
{data}

{LOG_WIDGET_HELP_TEXT}
</pre>
"""
    display(HTML(render_template(title, html_str)), clear=True)


def graceful_error_widget(error: str):
    """
    Render the widget with the graceful error and help.
    """

    title = "Oops!"
    html_str = f"""
<pre style="white-space: pre-wrap; word-break: break-word;">
{error}

Need help? Report your issue at https://feedback.haiqu.ai/.
You can also contact us on Slack: https://haiqu-community.slack.com/ or via support@haiqu.ai
</pre>
"""
    display(HTML(render_template(title, html_str)), clear=True)
