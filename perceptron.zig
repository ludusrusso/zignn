const std = @import("std");

pub const errs = error{PredictErr};

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

pub const Perceptron = struct {
    weights: []f64,
    bias: f64,
    act_fn: *const fn (x: f64) f64 = &relu,

    pub fn predict(self: *const Perceptron, xx: []const f64) errs!f64 {
        if (xx.len != self.weights.len) {
            // TODO: better errors
            return errs.PredictErr;
        }

        var res: f64 = self.bias;
        for (self.weights, xx) |w, x| {
            res += w * x;
        }

        return self.act_fn(res);
    }

    pub fn set_rand(self: *Perceptron) void {
        for (0..self.weights.len) |i| {
            self.weights[i] = 2 * std.crypto.random.float(f64) - 1.0;
        }
        self.bias = 2 * std.crypto.random.float(f64) - 1.0;
    }

    pub fn NewWithRelu(len: usize, init: f64) !Perceptron {
        const w = try allocator.alloc(f64, len);
        for (0..w.len) |i| {
            w[i] = init;
        }
        return Perceptron{
            .bias = 1,
            .weights = w,
            .act_fn = &relu,
        };
    }

    pub fn deinit(self: *Perceptron) void {
        allocator.free(self.weights);
    }
};

fn relu(x: f64) f64 {
    if (x < 0) {
        return 0;
    }

    return x;
}

const t = std.testing;

test "test perceptron 1" {
    const p = try Perceptron.NewWithRelu(3, 0);
    const input = [3]f64{ 0, 0, 0 };
    const res = try p.predict(&input);
    try t.expectEqual(1.0, res);
}

test "test perceptron 2" {
    const p = try Perceptron.NewWithRelu(3, 1);
    try t.expectEqual(4.0, try p.predict(&[3]f64{ 1, 1, 1 }));
    try t.expectEqual(0.0, try p.predict(&[3]f64{ 1, 1, -1000 }));
}

test "test perceptron 3 with Len = 2" {
    const p = try Perceptron.NewWithRelu(2, 1);

    try t.expectEqual(3.0, try p.predict(&[2]f64{ 1, 1 }));
    try t.expectEqual(0.0, try p.predict(&[2]f64{ 1, -1000 }));
}
