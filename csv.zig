const std = @import("std");
const DNN = @import("./dnn.zig");
const parseFloat = std.fmt.parseFloat;
const parseInt = std.fmt.parseInt;

pub fn main() !void {
    var file = try std.fs.cwd().openFile("./mnist_test.csv", .{});
    defer file.close();

    var dnn = try DNN.DNN.new(&[_]usize{ 784, 128, 64, 64, 10 });
    defer dnn.deinit();
    dnn.set_rand();

    var buf_reader = std.io.bufferedReader(file.reader());
    var in_stream = buf_reader.reader();

    var buf: [1024 * 4]u8 = undefined;
    while (try in_stream.readUntilDelimiterOrEof(&buf, '\n')) |line| {
        var it = std.mem.split(u8, line, ",");
        const num = it.next() orelse "";
        const y = try parseInt(u8, num, 10);
        var img = [_]f64{0} ** 784;
        var i: usize = 0;
        while (it.next()) |px| {
            img[i] = try parseFloat(f64, px) / 256.0;
            i += 1;
        }

        const res = try dnn.predict(&img);

        var id: usize = 0;
        var max = res[0];
        for (1..res.len) |ii| {
            if (res[ii] > max) {
                id = ii;
                max = res[ii];
            }
        }

        std.log.info("{d} -> {any}", .{ y, id });
    }
}
