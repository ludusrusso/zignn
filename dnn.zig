const std = @import("std");
const lnn = @import("./layer.zig");
const Layer = lnn.NNLayer;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

pub const DNN = struct {
    layers: []Layer,
    pub fn new(dims: []const usize) !DNN {
        var layers = try allocator.alloc(Layer, dims.len - 1);

        for (0..dims.len - 1, 1..dims.len) |i, j| {
            layers[i] = try Layer.New(dims[i], dims[j]);
        }

        return DNN{ .layers = layers };
    }

    pub fn predict(self: *const DNN, x: []const f64) ![]const f64 {
        var pred = x;
        for (self.layers) |l| {
            pred = try l.predict(pred);
        }

        return pred;
    }

    pub fn set_rand(self: *DNN) void {
        for (0..self.layers.len) |i| {
            self.layers[i].set_rand();
        }
    }

    pub fn deinit(self: *DNN) void {
        for (0..self.layers.len) |i| {
            self.layers[i].deinit();
        }

        allocator.free(self.layers);
    }
};

test "dnn allocation" {
    var dnn = try DNN.new(&[_]usize{ 3, 4, 2, 10 });
    defer dnn.deinit();

    const x = [_]f64{ 0, 0, 0 };

    const res = try dnn.predict(&x);

    std.log.err("output: {any}", .{res});

    try std.testing.expectEqual(10, res.len);
}

test "dnn allocation with random" {
    var dnn = try DNN.new(&[_]usize{ 3, 4, 2, 10 });
    defer dnn.deinit();
    dnn.set_rand();

    const x = [_]f64{ 0, 0, 0 };

    const res = try dnn.predict(&x);

    std.log.err("output: {any}", .{res});

    try std.testing.expectEqual(10, res.len);
}
