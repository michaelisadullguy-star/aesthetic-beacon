import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetSocketAddress;
import java.net.SocketException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * Minimal OSC-over-UDP receiver with no library dependencies.
 *
 * A background thread reads datagrams, decodes OSC messages (also inside
 * bundles) and puts them in a queue.  The sketch takes them out with poll()
 * at the start of each frame, so sketch state is only ever changed from the
 * animation thread.
 *
 * Supported argument types: int32 (i), float32 (f), string (s), int64 (h),
 * float64 (d), true/false (T/F).  Packets that cannot be decoded are dropped.
 */
public class OscReceiver implements Runnable {

  public static class Message {
    public final String address;
    public final Object[] args;

    Message(String address, Object[] args) {
      this.address = address;
      this.args = args;
    }
  }

  private final ConcurrentLinkedQueue<Message> queue = new ConcurrentLinkedQueue<Message>();
  private DatagramSocket socket;
  private volatile boolean running = false;
  private String error = null;
  private final int port;

  public OscReceiver(int port) {
    this.port = port;
    try {
      socket = new DatagramSocket(null);
      socket.setReuseAddress(true);
      socket.bind(new InetSocketAddress(port));
      running = true;
      Thread thread = new Thread(this, "osc-receiver");
      thread.setDaemon(true);
      thread.start();
    } catch (Exception e) {
      error = e.getMessage();
      socket = null;
    }
  }

  /** True while the UDP port is open. */
  public boolean isListening() {
    return running && socket != null;
  }

  /** Why the port could not be opened, or null. */
  public String getError() {
    return error;
  }

  public int getPort() {
    return port;
  }

  /** Next received message, or null when there is none. */
  public Message poll() {
    return queue.poll();
  }

  public void close() {
    running = false;
    if (socket != null) {
      socket.close();
    }
  }

  public void run() {
    byte[] buffer = new byte[65536];
    while (running) {
      DatagramPacket packet = new DatagramPacket(buffer, buffer.length);
      try {
        socket.receive(packet);
        parsePacket(buffer, 0, packet.getLength(), 0);
      } catch (SocketException e) {
        break; // socket closed
      } catch (Exception e) {
        // Malformed packet: ignore it and keep listening.
      }
    }
    running = false;
  }

  private void parsePacket(byte[] data, int offset, int length, int depth) {
    if (length >= 16 && data[offset] == '#' && new String(data, offset, 7, StandardCharsets.US_ASCII).equals("#bundle")) {
      if (depth > 8) {
        return;
      }
      int position = offset + 16; // "#bundle\0" + 8-byte time tag
      int end = offset + length;
      while (position + 4 <= end) {
        int size = ByteBuffer.wrap(data, position, 4).getInt();
        position += 4;
        if (size <= 0 || position + size > end) {
          return;
        }
        parsePacket(data, position, size, depth + 1);
        position += size;
      }
      return;
    }
    Message message = parseMessage(data, offset, length);
    if (message != null) {
      // Bound the backlog if the sketch is paused while messages keep arriving.
      if (queue.size() > 4096) {
        queue.poll();
      }
      queue.add(message);
    }
  }

  private static int stringEnd(byte[] data, int start, int end) {
    for (int i = start; i < end; i++) {
      if (data[i] == 0) {
        return i;
      }
    }
    return -1;
  }

  private static int padded(int length) {
    return (length / 4 + 1) * 4;
  }

  private static Message parseMessage(byte[] data, int offset, int length) {
    int end = offset + length;
    int addressEnd = stringEnd(data, offset, end);
    if (addressEnd < 0 || data[offset] != '/') {
      return null;
    }
    String address = new String(data, offset, addressEnd - offset, StandardCharsets.US_ASCII);
    int position = offset + padded(addressEnd - offset);
    if (position >= end) {
      return new Message(address, new Object[0]);
    }
    int tagsEnd = stringEnd(data, position, end);
    if (tagsEnd < 0 || data[position] != ',') {
      return null;
    }
    String tags = new String(data, position + 1, tagsEnd - position - 1, StandardCharsets.US_ASCII);
    position += padded(tagsEnd - position);

    ArrayList<Object> args = new ArrayList<Object>();
    for (int i = 0; i < tags.length(); i++) {
      char tag = tags.charAt(i);
      switch (tag) {
        case 'i':
          if (position + 4 > end) return null;
          args.add(Integer.valueOf(ByteBuffer.wrap(data, position, 4).getInt()));
          position += 4;
          break;
        case 'f':
          if (position + 4 > end) return null;
          args.add(Float.valueOf(ByteBuffer.wrap(data, position, 4).getFloat()));
          position += 4;
          break;
        case 'h':
          if (position + 8 > end) return null;
          args.add(Long.valueOf(ByteBuffer.wrap(data, position, 8).getLong()));
          position += 8;
          break;
        case 'd':
          if (position + 8 > end) return null;
          args.add(Double.valueOf(ByteBuffer.wrap(data, position, 8).getDouble()));
          position += 8;
          break;
        case 's':
          int stringStop = stringEnd(data, position, end);
          if (stringStop < 0) return null;
          args.add(new String(data, position, stringStop - position, StandardCharsets.UTF_8));
          position += padded(stringStop - position);
          break;
        case 'T':
          args.add(Boolean.TRUE);
          break;
        case 'F':
          args.add(Boolean.FALSE);
          break;
        case 'N':
        case 'I':
          break;
        default:
          return null; // unsupported type: drop the whole message
      }
    }
    return new Message(address, args.toArray());
  }

  /**
   * Stand-alone check: prints every received message as one line for the given
   * number of seconds.  Usage: java OscReceiver <port> <seconds>
   */
  public static void main(String[] arguments) throws Exception {
    int port = arguments.length > 0 ? Integer.parseInt(arguments[0]) : 12000;
    double seconds = arguments.length > 1 ? Double.parseDouble(arguments[1]) : 5.0;
    OscReceiver receiver = new OscReceiver(port);
    if (!receiver.isListening()) {
      System.out.println("ERROR " + receiver.getError());
      return;
    }
    System.out.println("LISTENING " + port);
    long deadline = System.currentTimeMillis() + (long) (seconds * 1000);
    while (System.currentTimeMillis() < deadline) {
      Message message = receiver.poll();
      if (message == null) {
        Thread.sleep(5);
        continue;
      }
      StringBuilder line = new StringBuilder(message.address);
      for (Object arg : message.args) {
        line.append('\t').append(arg.getClass().getSimpleName()).append(':').append(arg);
      }
      System.out.println(line);
    }
    receiver.close();
  }
}
