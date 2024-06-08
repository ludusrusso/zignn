const std = @import("std");
const perc = @import("./perceptron.zig");
const Perceptron = perc.Perceptron;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

pub const NNLayer = struct {
    perceptrons: []Perceptron,
    pub fn predict(self: *const NNLayer, x: []const f64) ![]f64 {
        var res = try allocator.alloc(f64, self.perceptrons.len);
        for (self.perceptrons, 0..) |p, i| {
            res[i] = try p.predict(x);
        }

        return res;
    }

    pub fn New(in_size: usize, out_size: usize) !NNLayer {
        const percs = try allocator.alloc(Perceptron, out_size);
        for (0..percs.len) |i| {
            percs[i] = try Perceptron.NewWithRelu(in_size, 1);
        }
        return NNLayer{ .perceptrons = percs };
    }

    pub fn set_rand(self: *NNLayer) void {
        for (0..self.perceptrons.len) |i| {
            self.perceptrons[i].set_rand();
        }
    }

    pub fn deinit(self: *NNLayer) void {
        for (self.perceptrons) |*p| {
            p.deinit();
        }

        allocator.free(self.perceptrons);
    }
};

test "nn_layers" {
    const l = try NNLayer.New(2, 2);
    const res = try l.predict(&[_]f64{ 0, 0 });

    try std.testing.expectEqual(2, res.len);
    for (res) |y| {
        try std.testing.expectEqual(1, y);
    }
}
