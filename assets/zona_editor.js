;(function () {
    'use strict';

    var ZE = {
        canvas:        null,
        ctx:           null,
        shapes:        [],
        dragType:      null,
        dragStart:     null,
        dragOrigShape: null,
        activeIdx:     -1,
    };

    var HANDLE    = 7;
    var MIN_SIZE  = 10;
    var MAX_SHAPES = 2;

    // BGR→RGB: norte (0,200,220)→rgb(220,200,0), sur (200,0,200)→rgb(200,0,200)
    var ZONE_COLORS = [
        { stroke: 'rgb(220,200,0)',   fill: 'rgba(220,200,0,0.12)',   handle: 'rgb(220,200,0)'   },
        { stroke: 'rgb(200,0,200)',   fill: 'rgba(200,0,200,0.12)',   handle: 'rgb(200,0,200)'   },
    ];
    var DEFAULT_COLOR = { stroke: '#888', fill: 'rgba(136,136,136,0.12)', handle: '#888' };

    function _color(i) { return ZONE_COLORS[i] || DEFAULT_COLOR; }

    // ── API pública ──────────────────────────────────────────────────

    window.zeInit = function (canvasId, initialShapes) {
        var c = document.getElementById(canvasId);
        if (!c) return;

        if (ZE.canvas !== c) {
            ZE.canvas = c;
            ZE.ctx    = c.getContext('2d');
            c.addEventListener('mousedown',  _onDown);
            c.addEventListener('mousemove',  _onMove);
            // mouseup y mousemove durante drag se gestionan en document
        }

        ZE.shapes    = (initialShapes || []).map(function (s) {
            return { x0: s.x0, y0: s.y0, x1: s.x1, y1: s.y1 };
        });
        ZE.dragType  = null;
        ZE.activeIdx = -1;
        _render();
        _updateCounter();
    };

    // ── Posición ─────────────────────────────────────────────────────

    function _rawPos(e) {
        var r = ZE.canvas.getBoundingClientRect();
        return {
            x: (e.clientX - r.left) * ZE.canvas.width  / r.width,
            y: (e.clientY - r.top)  * ZE.canvas.height / r.height
        };
    }

    function _clampPos(e) {
        var r = ZE.canvas.getBoundingClientRect();
        return {
            x: Math.max(0, Math.min(ZE.canvas.width,  (e.clientX - r.left) * ZE.canvas.width  / r.width)),
            y: Math.max(0, Math.min(ZE.canvas.height, (e.clientY - r.top)  * ZE.canvas.height / r.height))
        };
    }

    // ── Hit-testing ──────────────────────────────────────────────────

    function _hitTest(s, x, y) {
        var mx = (s.x0 + s.x1) / 2, my = (s.y0 + s.y1) / 2;
        var handles = [
            { n: 'nw', x: s.x0, y: s.y0 }, { n: 'n', x: mx,   y: s.y0 }, { n: 'ne', x: s.x1, y: s.y0 },
            { n: 'w',  x: s.x0, y: my   },                                  { n: 'e',  x: s.x1, y: my   },
            { n: 'sw', x: s.x0, y: s.y1 }, { n: 's', x: mx,   y: s.y1 }, { n: 'se', x: s.x1, y: s.y1 },
        ];
        for (var i = 0; i < handles.length; i++) {
            var h = handles[i];
            if (Math.abs(x - h.x) <= HANDLE && Math.abs(y - h.y) <= HANDLE)
                return 'resize-' + h.n;
        }
        if (x > s.x0 + HANDLE && x < s.x1 - HANDLE &&
            y > s.y0 + HANDLE && y < s.y1 - HANDLE)
            return 'move';
        return null;
    }

    function _cursorFor(hit) {
        if (!hit || hit === 'draw') return 'crosshair';
        if (hit === 'move')         return 'move';
        if (hit === 'resize-nw' || hit === 'resize-se') return 'nwse-resize';
        if (hit === 'resize-ne' || hit === 'resize-sw') return 'nesw-resize';
        if (hit === 'resize-n'  || hit === 'resize-s')  return 'ns-resize';
        if (hit === 'resize-e'  || hit === 'resize-w')  return 'ew-resize';
        return 'crosshair';
    }

    // ── Eventos de ratón ─────────────────────────────────────────────

    function _onDown(e) {
        if (e.button !== 0) return;
        var p = _rawPos(e);
        var started = false;

        // ¿Hit sobre forma existente?
        for (var i = ZE.shapes.length - 1; i >= 0; i--) {
            var hit = _hitTest(ZE.shapes[i], p.x, p.y);
            if (hit) {
                ZE.activeIdx     = i;
                ZE.dragType      = hit;
                ZE.dragStart     = p;
                ZE.dragOrigShape = { x0: ZE.shapes[i].x0, y0: ZE.shapes[i].y0,
                                     x1: ZE.shapes[i].x1, y1: ZE.shapes[i].y1 };
                ZE.canvas.style.cursor = _cursorFor(hit);
                started = true;
                break;
            }
        }

        // Dibujar nueva forma
        if (!started && ZE.shapes.length < MAX_SHAPES) {
            ZE.activeIdx = ZE.shapes.length;
            ZE.dragType  = 'draw';
            ZE.dragStart = p;
            ZE.shapes.push({ x0: p.x, y0: p.y, x1: p.x, y1: p.y });
            ZE.canvas.style.cursor = 'crosshair';
            started = true;
        }

        if (started) {
            // Escuchar en document para que el drag continúe fuera del canvas
            document.addEventListener('mousemove', _onDocMove);
            document.addEventListener('mouseup',   _onDocUp);
            e.preventDefault();
            _render();
        }
    }

    function _onMove(e) {
        // Solo actualiza cursor cuando no hay drag en curso
        if (ZE.dragType || !ZE.canvas) return;
        var p = _rawPos(e);
        for (var i = ZE.shapes.length - 1; i >= 0; i--) {
            var hit = _hitTest(ZE.shapes[i], p.x, p.y);
            if (hit) { ZE.canvas.style.cursor = _cursorFor(hit); return; }
        }
        ZE.canvas.style.cursor = 'crosshair';
    }

    function _onDocMove(e) {
        if (!ZE.dragType || !ZE.canvas) return;
        var p    = _clampPos(e);           // coordenadas limitadas al borde del canvas
        var dx   = p.x - ZE.dragStart.x;
        var dy   = p.y - ZE.dragStart.y;
        var orig = ZE.dragOrigShape;
        var otro = ZE.shapes.length > 1 ? ZE.shapes[1 - ZE.activeIdx] : null;

        if (ZE.dragType === 'draw') {
            var ns = {
                x0: Math.min(ZE.dragStart.x, p.x),
                y0: Math.min(ZE.dragStart.y, p.y),
                x1: Math.max(ZE.dragStart.x, p.x),
                y1: Math.max(ZE.dragStart.y, p.y),
            };
            ZE.shapes[ZE.activeIdx] = _sinSolape(ns, otro);

        } else if (ZE.dragType === 'move') {
            var w = orig.x1 - orig.x0, h = orig.y1 - orig.y0;
            var nx0 = Math.max(0, Math.min(ZE.canvas.width  - w, orig.x0 + dx));
            var ny0 = Math.max(0, Math.min(ZE.canvas.height - h, orig.y0 + dy));
            ZE.shapes[ZE.activeIdx] = _sinSolapeMover(
                { x0: nx0, y0: ny0, x1: nx0 + w, y1: ny0 + h }, otro
            );

        } else {
            // Resize: cada arista se mueve de forma independiente
            // sin cruzar la arista opuesta (sin invertir el cuadrado)
            var dir = ZE.dragType.replace('resize-', '');
            var s = { x0: orig.x0, y0: orig.y0, x1: orig.x1, y1: orig.y1 };
            if (dir.indexOf('w') >= 0)
                s.x0 = Math.max(0,               Math.min(orig.x0 + dx, orig.x1 - MIN_SIZE));
            if (dir.indexOf('e') >= 0)
                s.x1 = Math.min(ZE.canvas.width,  Math.max(orig.x1 + dx, orig.x0 + MIN_SIZE));
            if (dir.indexOf('n') >= 0)
                s.y0 = Math.max(0,               Math.min(orig.y0 + dy, orig.y1 - MIN_SIZE));
            if (dir.indexOf('s') >= 0)
                s.y1 = Math.min(ZE.canvas.height, Math.max(orig.y1 + dy, orig.y0 + MIN_SIZE));
            ZE.shapes[ZE.activeIdx] = _sinSolape(s, otro);
        }

        _render();
    }

    function _onDocUp() {
        document.removeEventListener('mousemove', _onDocMove);
        document.removeEventListener('mouseup',   _onDocUp);
        if (!ZE.dragType) return;

        // Eliminar formas demasiado pequeñas
        ZE.shapes = ZE.shapes.filter(function (s) {
            return (s.x1 - s.x0) >= MIN_SIZE && (s.y1 - s.y0) >= MIN_SIZE;
        });
        if (ZE.activeIdx >= ZE.shapes.length) ZE.activeIdx = -1;
        ZE.dragType      = null;
        ZE.dragStart     = null;
        ZE.dragOrigShape = null;
        if (ZE.canvas) ZE.canvas.style.cursor = 'crosshair';

        _render();
        _notifyDash();
        _updateCounter();
    }

    // ── Render ───────────────────────────────────────────────────────

    function _render() {
        var ctx = ZE.ctx;
        if (!ctx) return;
        ctx.clearRect(0, 0, ZE.canvas.width, ZE.canvas.height);

        ZE.shapes.forEach(function (s, i) {
            var c = _color(i);
            var w = s.x1 - s.x0, h = s.y1 - s.y0;

            ctx.fillStyle   = c.fill;
            ctx.fillRect(s.x0, s.y0, w, h);
            ctx.strokeStyle = c.stroke;
            ctx.lineWidth   = 2;
            ctx.strokeRect(s.x0, s.y0, w, h);

            _drawHandles(s, c.handle);
        });
    }

    function _drawHandles(s, color) {
        var ctx = ZE.ctx;
        var mx  = (s.x0 + s.x1) / 2, my = (s.y0 + s.y1) / 2;
        var pts = [
            [s.x0, s.y0], [mx, s.y0], [s.x1, s.y0],
            [s.x0, my  ],             [s.x1, my  ],
            [s.x0, s.y1], [mx, s.y1], [s.x1, s.y1],
        ];
        ctx.fillStyle   = color;
        ctx.strokeStyle = '#1a1a2e';
        ctx.lineWidth   = 1;
        pts.forEach(function (pt) {
            ctx.fillRect  (pt[0] - HANDLE, pt[1] - HANDLE, HANDLE * 2, HANDLE * 2);
            ctx.strokeRect(pt[0] - HANDLE, pt[1] - HANDLE, HANDLE * 2, HANDLE * 2);
        });
    }

    // ── Anti-solapamiento ────────────────────────────────────────────

    // Devuelve s ajustado para no solapar con otro, pudiendo cambiar tamaño
    // (uso: draw y resize — el borde que invade se retrae a la arista de otro).
    function _sinSolape(s, otro) {
        if (!otro) return s;
        if (s.x1 <= otro.x0 || s.x0 >= otro.x1 || s.y1 <= otro.y0 || s.y0 >= otro.y1) return s;
        var ops = [
            { pen: s.x1 - otro.x0, r: { x0: s.x0,   y0: s.y0, x1: otro.x0, y1: s.y1   } },
            { pen: otro.x1 - s.x0, r: { x0: otro.x1, y0: s.y0, x1: s.x1,   y1: s.y1   } },
            { pen: s.y1 - otro.y0, r: { x0: s.x0, y0: s.y0,   x1: s.x1, y1: otro.y0   } },
            { pen: otro.y1 - s.y0, r: { x0: s.x0, y0: otro.y1, x1: s.x1, y1: s.y1     } },
        ];
        var mejor = ops.reduce(function(a, b) { return a.pen <= b.pen ? a : b; }).r;
        if ((mejor.x1 - mejor.x0) < MIN_SIZE || (mejor.y1 - mejor.y0) < MIN_SIZE) return s;
        return mejor;
    }

    // Devuelve s ajustado para no solapar con otro, conservando tamaño
    // (uso: move — la caja se detiene al llegar al borde de otro).
    function _sinSolapeMover(s, otro) {
        if (!otro) return s;
        if (s.x1 <= otro.x0 || s.x0 >= otro.x1 || s.y1 <= otro.y0 || s.y0 >= otro.y1) return s;
        var w = s.x1 - s.x0, h = s.y1 - s.y0;
        var ops = [
            { pen: s.x1 - otro.x0, r: { x0: otro.x0 - w, y0: s.y0, x1: otro.x0,     y1: s.y1     } },
            { pen: otro.x1 - s.x0, r: { x0: otro.x1,     y0: s.y0, x1: otro.x1 + w, y1: s.y1     } },
            { pen: s.y1 - otro.y0, r: { x0: s.x0, y0: otro.y0 - h, x1: s.x1,        y1: otro.y0  } },
            { pen: otro.y1 - s.y0, r: { x0: s.x0, y0: otro.y1,     x1: s.x1,        y1: otro.y1 + h } },
        ];
        return ops.reduce(function(a, b) { return a.pen <= b.pen ? a : b; }).r;
    }

    // ── Notificaciones ───────────────────────────────────────────────

    function _notifyDash() {
        if (window.dash_clientside && window.dash_clientside.set_props) {
            window.dash_clientside.set_props('zona-editor-formas', {
                data: ZE.shapes.map(function (s) {
                    return { x0: s.x0, y0: s.y0, x1: s.x1, y1: s.y1 };
                })
            });
        }
    }

    function _updateCounter() {
        var n      = ZE.shapes.length;
        var colors = ['#607d8b', '#ff9800', '#00e676'];
        var el     = document.getElementById('zona-contador');
        if (!el) return;
        el.textContent = n + '/2 zonas dibujadas';
        el.style.color = colors[Math.min(n, 2)];
    }

}());
