(use-modules (grand scheme)
	     ((rnrs) #:version (6) #:select (make-bytevector
					     utf8->string string->utf8
					     bytevector-fill!)))

(load-extension "./clops" "init")

(let* ((source "__kernel void id(__global char *out) {
  const uint i = get_global_id(0);
  out[i] = 255 - (char) (i & 255);
}")
       (`(,platform . ,_) (cl-platforms))
       (`(,gpu) (cl-devices platform 'GPU))
       (size 256)
       (result (make-bytevector size)))
  (call-with-cl-context
   (cl-make-context gpu)
   (lambda ()
     (let* ((command-queue (cl-make-command-queue gpu))
	    (program (cl-make-program source gpu))
	    (kernel (cl-kernel program "id"))
	    (out (cl-make-buffer result 'read/write 'use-host-pointer)))
       (cl-bind-arguments kernel out)
       (cl-enqueue-kernel! command-queue kernel size)
       (cl-enqueue-read-buffer! command-queue out)
       (cl-finish! command-queue)
       result
       ))))
