#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <alloca.h>

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/opencl.h>

#include <libguile.h>

#define NELEMS(array) (sizeof(array)/sizeof(array[0]))

#define WARN_(msg, ...) fprintf(stderr, msg, ## __VA_ARGS__)
#define WARN(msg, ...) fprintf(stderr, msg "\n", ## __VA_ARGS__)

#define DUMP(expression, format) WARN(# expression ": "format, expression)

static scm_t_bits cl_platform_tag;
static scm_t_bits cl_device_tag;
static scm_t_bits cl_context_tag;
static scm_t_bits cl_command_queue_tag;
static scm_t_bits cl_program_tag;
static scm_t_bits cl_kernel_tag;
static scm_t_bits cl_buffer_tag;
static scm_t_bits cl_sampler_tag;
static scm_t_bits cl_image2d_tag;
static scm_t_bits cl_image3d_tag;
static scm_t_bits cl_event_tag;

#define CL_TRY(action) if((action) != CL_SUCCESS) { \
    WARN(# action " failed");			    \
    return SCM_BOOL_F;				    \
  }

static void
cl_warn(cl_int result) {
  switch(result) {
  case CL_INVALID_PROGRAM_EXECUTABLE:
    WARN("invalid program executable");
    break;
  case CL_INVALID_COMMAND_QUEUE:
    WARN("invalid command queue");
    break;
  case CL_INVALID_KERNEL:
    WARN("invalid kernel");
    break;
  case CL_INVALID_KERNEL_ARGS:
    WARN("invalid kernel argument");
    break;
  case CL_INVALID_ARG_INDEX:
    WARN("invalid argument index");
    break;
  case CL_INVALID_ARG_VALUE:
    WARN("invalid argument value");
    break;
  case CL_INVALID_SAMPLER:
    WARN("invalid sampler");
    break;
  case CL_INVALID_ARG_SIZE:
    WARN("invalid argument size");
    break;
  case CL_INVALID_MEM_OBJECT:
    WARN("invalid memory object");
    break;
  case CL_INVALID_WORK_DIMENSION:
    WARN("invalid work dimension");
    break;
  case CL_INVALID_WORK_GROUP_SIZE:
    WARN("invalid work group size");
    break;
  case CL_INVALID_WORK_ITEM_SIZE:
    WARN("invalid work item size");
    break;
  case CL_INVALID_GLOBAL_OFFSET:
    WARN("invalid global offset");
    break;
  case CL_OUT_OF_RESOURCES:
    WARN("out of resources");
    break;
  case CL_INVALID_EVENT_WAIT_LIST:
    WARN("invalid event wait list");
    break;
  case CL_INVALID_CONTEXT:
    WARN("invalid context");
    break;
  case CL_INVALID_VALUE:
    WARN("invalid value");
    break;
  case CL_INVALID_BUFFER_SIZE:
    WARN("invalid buffer size");
    break;
  case CL_INVALID_HOST_PTR:
    WARN("invalid host pointer (did you forget the copy/use host pointer flag?)");
    break;
  case CL_MEM_OBJECT_ALLOCATION_FAILURE:
    WARN("memory object allocation failure");
    break;
  case CL_OUT_OF_HOST_MEMORY:
    WARN("out of host memory");
    break;
  default:
    WARN("unknown error: 0x%x", result);
    break;
  }
}

static SCM
platform_smob(cl_platform_id platform_id) {
  assert(sizeof(scm_t_bits) == sizeof(cl_platform_id));
  SCM platform = scm_new_smob(cl_platform_tag, (scm_t_bits) platform_id);
  return platform;
}

static const char *
platform_param_x(cl_platform_id platform_id,
		 cl_platform_info param,
		 const char *fmt)
{
  static char buffer[64];
  size_t size;
  if(clGetPlatformInfo(platform_id, param, sizeof(buffer), buffer, &size)
     == CL_SUCCESS) {
    assert(sizeof(buffer) >= size);
    if(!strcmp("%d", fmt) || !strcmp("%x", fmt)) {
      int value = *((int *) buffer);
      snprintf(buffer, sizeof(buffer), fmt, value);
    }
    else if(strcmp("%s", fmt)) {
      WARN("Unsupported format: %s", fmt);
    }
    else {
      // the presence of "%s" is the expected option,
      // and it means that we don't need to perform
      // any conversions
    }
  }
  else {
    snprintf(buffer, sizeof(buffer), "???");
  }
  return (const char *) buffer;
}

static int
platform_smob_print(SCM platform, SCM port, scm_print_state *unused) {
  cl_platform_id id = (cl_platform_id) SCM_SMOB_DATA(platform);
  char buffer[16];
  scm_puts("#<OpenCL platform ", port);
  snprintf(buffer, sizeof(buffer), "%x ", (void *) id);
  scm_puts(buffer, port);
  scm_puts(platform_param_x(id, CL_PLATFORM_NAME, "%s"), port);
  scm_puts(" ", port);
  scm_puts(platform_param_x(id, CL_PLATFORM_PROFILE, "%s"), port);
  scm_puts(" ", port);
  scm_puts(platform_param_x(id, CL_PLATFORM_VERSION, "%s"), port);
  scm_puts(">", port);
}

static SCM
platforms() {
  cl_uint num_platforms;
  SCM result = SCM_EOL;
  if(clGetPlatformIDs(0, NULL, &num_platforms) == CL_SUCCESS) {
    cl_platform_id *platforms = alloca(num_platforms * sizeof(cl_platform_id));
    if(clGetPlatformIDs(num_platforms, platforms, NULL) == CL_SUCCESS) {
      for(int i = 0; i < num_platforms; ++i) {
	result = scm_cons(platform_smob(platforms[i]), result);
      }
    } else {
      WARN("clGetPlatformIDs(%d, %x, NULL) failed", num_platforms, platforms);
      result = SCM_BOOL_F;
    }
  }
  return result;
}

static cl_device_type
parse_device_types(SCM types)
{
  //SCM_ASSERT_TYPE(scm_is_list(types), types, SCM_ARG1, __FUNCTION__, "list");
  cl_device_type device_type = (cl_device_type) 0;
  if(scm_is_null(types)) {
    device_type = CL_DEVICE_TYPE_ALL;
  }
  else {
    for(; scm_is_pair(types); types = scm_cdr(types)) {
      char *symbol = scm_to_locale_string(scm_symbol_to_string(scm_car(types)));
      if(!strcasecmp("GPU", symbol)) {
	device_type |= CL_DEVICE_TYPE_GPU;
      }
      else if(!strcasecmp("CPU", symbol)) {
	device_type |= CL_DEVICE_TYPE_CPU;
      }
      else if(!strcasecmp("ACCELERATOR", symbol)) {
	device_type |= CL_DEVICE_TYPE_ACCELERATOR;	
      }
      else if(!strcasecmp("DEFAULT", symbol)) {
	device_type |= CL_DEVICE_TYPE_DEFAULT;
      }
      else if(!strcasecmp("CUSTOM", symbol)) {
	device_type |= CL_DEVICE_TYPE_CUSTOM;
      }
      else if(!strcasecmp("ALL", symbol)) {
	device_type |= CL_DEVICE_TYPE_ALL;
      }
      else {
	WARN("Unsupported device type: %s", symbol);
      }
      free(symbol);
    }
  }
  return device_type;
}

static SCM
device_smob(cl_device_id device_id, cl_platform_id platform_id)
{
  assert(sizeof(scm_t_bits) == sizeof(cl_platform_id)
	 && sizeof(scm_t_bits) == sizeof(cl_device_id));
  SCM device = scm_new_double_smob(cl_device_tag,
				   (scm_t_bits) device_id,
				   (scm_t_bits) platform_id,
				   (scm_t_bits) NULL);
  return device;
}

static void
as_string(char *buffer, size_t length, const char *fmt)
{
  if(!strcmp("%d", fmt) || !strcmp("%x", fmt)) {
    int value = *((int *) buffer);
    snprintf(buffer, length, fmt, value);
  }
  else if(!strcasecmp("device-type", fmt)) {
    assert(length >= sizeof(cl_device_type));
    cl_device_type device_type = *((cl_device_type *) buffer);
    switch(device_type) {
    case CL_DEVICE_TYPE_GPU:
      snprintf(buffer, length, "GPU");
      break;
    case CL_DEVICE_TYPE_CPU:
      snprintf(buffer, length, "CPU");
      break;
    case CL_DEVICE_TYPE_ACCELERATOR:
      snprintf(buffer, length, "ACCELERATOR");
      break;
    case CL_DEVICE_TYPE_CUSTOM:
      snprintf(buffer, length, "CUSTOM");
      break;
    case CL_DEVICE_TYPE_DEFAULT:
      snprintf(buffer, length, "DEFAULT");
      break;
    default:
      snprintf(buffer, length, "UNKNOWN");
      break;
    }
  }
  else {
    WARN("Unsupported format: %s", fmt);
  }
}

static const char*
device_param_x(cl_device_id device_id,
	       cl_device_info param,
	       const char *fmt)
{
  static char buffer[64];
  size_t size;
  if(clGetDeviceInfo(device_id, param, sizeof(buffer), buffer, &size)
     == CL_SUCCESS) {
    assert(sizeof(buffer) >= size);
    if(fmt != NULL && strcmp("%s", fmt) && strcmp("", fmt)) {
      as_string(buffer, sizeof(buffer), fmt);
    }
  }
  else {
    snprintf(buffer, sizeof(buffer), "???");
  }
  return (const char *) buffer;
}

static int
device_smob_print(SCM device, SCM port, scm_print_state *unused) {
  cl_device_id device_id = (cl_device_id) SCM_SMOB_DATA(device);
  cl_platform_id platform_id = (cl_platform_id) SCM_SMOB_DATA_2(device);
  char buffer[16];
  scm_puts("#<OpenCL device ", port);
  snprintf(buffer, sizeof(buffer), "%x ", (void *) device_id);
  scm_puts(buffer, port);
  scm_puts(device_param_x(device_id, CL_DEVICE_TYPE, "device-type"), port);
  scm_puts(" ", port);
  scm_puts(device_param_x(device_id, CL_DEVICE_VENDOR, "%s"), port);
  scm_puts(" ", port);
  scm_puts(device_param_x(device_id, CL_DEVICE_VERSION, "%s"), port);
  scm_puts(" ", port);
  scm_puts(device_param_x(device_id, CL_DRIVER_VERSION, "%s"), port);
  scm_puts(">", port); 
}

static SCM
devices(SCM platform, SCM types) {
  scm_assert_smob_type(cl_platform_tag, platform);
  cl_platform_id platform_id = (cl_platform_id) SCM_SMOB_DATA(platform);
  cl_uint num_devices;
  SCM result = SCM_EOL;
  cl_device_type device_type = parse_device_types(types);
  if(clGetDeviceIDs(platform_id, device_type, 0, NULL, &num_devices)
     == CL_SUCCESS) {
    cl_device_id *devices = alloca(num_devices * sizeof(cl_device_id));
    if(clGetDeviceIDs(platform_id, device_type, num_devices,
		      devices, NULL) == CL_SUCCESS) {
      for(int i = 0; i < num_devices; ++i) {
	result = scm_cons(device_smob(devices[i], platform_id), result);
      }
    }
    else {
      WARN("clGetDeviceIDs(%x, %x, %d, %x, NULL) failed",
	   platform_id, device_type, num_devices, devices);
      result = SCM_BOOL_F;
    }
  }
  return result;
}

static void
on_error_in_context(const char *errinfo, const void *private_info,
		    size_t cb, void *user_data)
{
  WARN("Error: %s", errinfo);
}

static SCM context_stack = SCM_EOL;

static inline SCM
current_context() {
  return scm_car(context_stack);
}

  
static SCM
call_with_context(SCM context, SCM thunk) {
  scm_assert_smob_type(cl_context_tag, context);
  context_stack = scm_cons(context, context_stack);
  SCM result = scm_call_0(thunk);
  context_stack = scm_cdr(context_stack);
  return result;
}

static SCM
set_current_context_x(SCM context) {
  scm_assert_smob_type(cl_context_tag, context);
  if(scm_is_null(context_stack)) {
    context_stack = scm_cons(context, context_stack);
  }
  else {
    scm_set_car_x(context_stack, context);
  }
  return SCM_UNSPECIFIED;
}

static SCM
create_context(SCM device_smobs) {
  if(scm_is_null(device_smobs)) {
    WARN("No devices for context");
    return SCM_BOOL_F;
  }
  SCM first_device = scm_car(device_smobs);
  scm_assert_smob_type(cl_device_tag, first_device);
  cl_platform_id platform = (cl_platform_id) SCM_SMOB_DATA_2(first_device);
  int num_devices = 1;
  
  for(SCM d = scm_cdr(device_smobs); scm_is_pair(d); d = scm_cdr(d)) {
    SCM device = scm_car(d);
    scm_assert_smob_type(cl_device_tag, device);
    ++num_devices;
    if(((cl_platform_id) SCM_SMOB_DATA_2(device)) != platform) {
      WARN("Requested context for devices from different platforms");
      return SCM_BOOL_F;
    }
  }

  cl_device_id *devices = alloca(num_devices * sizeof(cl_device_id));
  for(int i = 0;
      scm_is_pair(device_smobs);
      ++i, device_smobs = scm_cdr(device_smobs)) {
    devices[i] = (cl_device_id) SCM_SMOB_DATA(scm_car(device_smobs));
  }
  
  cl_context_properties properties[] = {
    CL_CONTEXT_PLATFORM, (cl_context_properties) platform, 0
  };

  SCM context_smob;
  cl_int result;
  cl_context c = clCreateContext(properties, num_devices, devices,
				 on_error_in_context, NULL, &result);
  if(result == CL_SUCCESS) {
    context_smob = scm_new_smob(cl_context_tag, (scm_t_bits) c);
  }
  else {
    WARN("Failed to create context (0x%x)", result);
    context_smob = SCM_BOOL_F;
  }
  return context_smob;
}

static cl_command_queue_properties
parse_command_queue_properties(SCM properties) {
  cl_command_queue_properties result = (cl_command_queue_properties) 0;
  for(; scm_is_pair(properties); properties = scm_cdr(properties)) {
    char *symbol
      = scm_to_locale_string(scm_symbol_to_string(scm_car(properties)));
    if(!strcasecmp("out-of-order-execution-mode", symbol)) {
      result |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    }
    else if(!strcasecmp("profiling", symbol)) {
      result |= CL_QUEUE_PROFILING_ENABLE;
    }
    else {
      WARN("Unsupported command queue property: %s", symbol);
    }
    free(symbol);
  }
  return result;
}

static char *current_build_options = "";

static SCM call_with_build_options(SCM options, SCM thunk) {
  char *previous_build_options = current_build_options;
  current_build_options = scm_to_locale_string(options);
  SCM result = scm_call_0(thunk);
  free(current_build_options);
  current_build_options = previous_build_options;
  return result;
}

static SCM
create_command_queue(SCM device, SCM properties) {
  assert(sizeof(cl_command_queue) == sizeof(scm_t_bits));
  scm_assert_smob_type(cl_device_tag, device);
  cl_context context = (cl_context) SCM_SMOB_DATA(current_context());
  cl_device_id device_id = (cl_device_id) SCM_SMOB_DATA(device);
  cl_command_queue_properties props = parse_command_queue_properties(properties);
  SCM queue;
  cl_int result;
  cl_command_queue q = clCreateCommandQueue(context, device_id, props, &result);
  if(result == CL_SUCCESS) {
    queue = scm_new_smob(cl_command_queue_tag, (scm_t_bits) q);
  }
  else {
    WARN("Failed to create command queue (0x%x)", result);
    queue = SCM_BOOL_F;
  }
  
  return queue;
}



static SCM
create_program(SCM source, SCM devices) {
  assert(sizeof(cl_program) == sizeof(scm_t_bits));
  char *src = scm_to_locale_string(source);
  const char *prog[] = { src };
  cl_context context = (cl_context) SCM_SMOB_DATA(current_context());
  cl_int result;
  SCM program;
  cl_program handle
    = clCreateProgramWithSource(context, NELEMS(prog), prog, NULL, &result);
  if(result == CL_SUCCESS) {
    program = scm_new_smob(cl_program_tag, (scm_t_bits) handle);
    cl_device_id *device_ids = NULL;
    int num_devices = scm_to_int(scm_length(devices));
    if(scm_is_pair(devices)) {
      device_ids = alloca(num_devices * sizeof(cl_device_id));
      for(int i = 0; scm_is_pair(devices); ++i, devices = scm_cdr(devices)) {
	SCM dev = scm_car(devices);
	scm_assert_smob_type(cl_device_tag, dev);
	device_ids[i] = (cl_device_id) SCM_SMOB_DATA(dev);
      }
    }
    result = clBuildProgram(handle, num_devices, device_ids,
			    current_build_options,
			    (void (*)(cl_program, void *)) NULL,
			    NULL);
    if(result != CL_SUCCESS) {
      WARN("Failed to build program (0x%x)", result); 
    }
  }
  else {
    WARN("Failed to create program (0x%x)", result);
    program = SCM_BOOL_F;
  }
  free(src);
  return program;
}

static SCM
kernel(SCM program, SCM name) {
  assert(sizeof(cl_kernel) == sizeof(scm_t_bits));
  scm_assert_smob_type(cl_program_tag, program);
  char *fname = scm_to_locale_string(name);
  cl_program program_id = (cl_program) SCM_SMOB_DATA(program);
  cl_int result;
  SCM kernel;
  cl_kernel kernel_id = clCreateKernel(program_id, fname, &result);
  if(result == CL_SUCCESS) {
    kernel = scm_new_double_smob(cl_kernel_tag,
				 (scm_t_bits) kernel_id,
				 (scm_t_bits) fname,
				 (scm_t_bits) NULL);
  }
  else {
    WARN("Failed to create kernel %s for program %llx: 0x%x", fname,
	 kernel_id, result);
    kernel = SCM_BOOL_F;
    free(fname);
  }
  return kernel;
}

static int
kernel_smob_print(SCM kernel, SCM port, scm_print_state *unused) {
  char *name = (char *) SCM_SMOB_DATA_2(kernel);
  scm_puts("#<OpenCL kernel ", port);
  scm_puts(name, port);
  scm_puts(">", port);
}

static size_t
kernel_smob_free(SCM kernel) {
  void *name = (void *) SCM_SMOB_DATA_2(kernel);
  free(name);
  return 0;
}

cl_mem_flags parse_mem_flags(SCM symbols) {
  cl_mem_flags flags = (cl_mem_flags) 0;
  for(; scm_is_pair(symbols); symbols = scm_cdr(symbols)) {
    char *symbol
      = scm_to_locale_string(scm_symbol_to_string(scm_car(symbols)));
    if(!strcasecmp("read-write", symbol)
       || !strcasecmp("read_write", symbol)
       || !strcasecmp("read/write", symbol)) {
      flags |= CL_MEM_READ_WRITE;
    }
    else if (!strcasecmp("read-only", symbol)
	     || !strcasecmp("read_only", symbol)) {
      flags |= CL_MEM_READ_ONLY;
    }
    else if (!strcasecmp("write-only", symbol)
	     || !strcasecmp("write_only", symbol)) {
      flags |= CL_MEM_WRITE_ONLY;
    }
    else if (!strcasecmp("use-host-pointer", symbol)
	     || !strcasecmp("use_host_pointer", symbol)
	     || !strcasecmp("use-host-ptr", symbol)
	     || !strcasecmp("use_host_ptr", symbol)) {
      flags |= CL_MEM_USE_HOST_PTR;
    }
    else if (!strcasecmp("allocate-host-pointer", symbol)
	     || !strcasecmp("allocate_host_pointer", symbol)
	     || !strcasecmp("alloc-host-ptr", symbol)
	     || !strcasecmp("alloc_host_ptr", symbol)) {
      flags |= CL_MEM_ALLOC_HOST_PTR;
    }
    else if (!strcasecmp("copy-host-pointer", symbol)
	     || !strcasecmp("copy_host_pointer", symbol)
	     || !strcasecmp("copy-host-ptr", symbol)
	     || !strcasecmp("copy_host_ptr", symbol)) {
      flags |= CL_MEM_COPY_HOST_PTR;      
    }
    else {
      WARN("Unsupported buffer creation option: %s", symbol);
    }
    
    free(symbol);
  }
  return flags;
}


static SCM
create_buffer(SCM source, SCM options) {
  assert(sizeof(cl_mem) == sizeof(scm_t_bits));
  
  size_t size = 0;
  void *host_ptr = NULL;
  if(scm_is_integer(source)) {
    size = scm_to_size_t(source);
  }
  else if(scm_is_bytevector(source)) {
    size = scm_c_bytevector_length(source);
    host_ptr = SCM_BYTEVECTOR_CONTENTS(source);
  }
  else {
    WARN("Unsupported source type");
    return SCM_BOOL_F;
  }
  cl_mem_flags flags = parse_mem_flags(options);
  if(flags == (cl_mem_flags) 0) {
    flags = host_ptr ? CL_MEM_USE_HOST_PTR : CL_MEM_READ_WRITE;
  }
  cl_context context = (cl_context) SCM_SMOB_DATA(current_context());
  cl_int result;
  cl_mem buffer = clCreateBuffer(context, flags, size, host_ptr, &result);
  SCM buffer_smob;
  
  if(result == CL_SUCCESS) {
    buffer_smob = scm_new_double_smob(cl_buffer_tag,
				      (scm_t_bits) buffer,
				      (scm_t_bits) size,
				      (scm_t_bits) host_ptr);
  }
  else {
    WARN_("Failed to initialize buffer of size %d: ", size);
    cl_warn(result);
    buffer_smob = SCM_BOOL_F;
  }
  return buffer_smob;
}

static SCM
bind_arguments(SCM kernel, SCM arguments) {
  scm_assert_smob_type(cl_kernel_tag, kernel);
  cl_kernel kernel_id = (cl_kernel) SCM_SMOB_DATA(kernel);
  char *kernel_name = (char *) SCM_SMOB_DATA_2(kernel);
  
  for(int i = 0; scm_is_pair(arguments); ++i, arguments = scm_cdr(arguments)) {
    SCM argument = scm_car(arguments);
    cl_int result;
    cl_mem buffer = (cl_mem) SCM_SMOB_DATA(argument);
    if(SCM_SMOB_PREDICATE(cl_buffer_tag, argument)
       || SCM_SMOB_PREDICATE(cl_sampler_tag, argument)
       || SCM_SMOB_PREDICATE(cl_image2d_tag, argument)
       || SCM_SMOB_PREDICATE(cl_image3d_tag, argument)) {
      result = clSetKernelArg(kernel_id, i, sizeof(cl_mem), &buffer);
    }
    else {
      WARN("Unrecognized argument type for argument %d to kernel %s",
	   i, kernel_name);
      result = clSetKernelArg(kernel_id, i, 0, NULL);
    }
    if(result != CL_SUCCESS) {
      WARN_("Binding argument %d to kernel %s failed: ", i, kernel_name);
      cl_warn(result);
    }
  }
  return SCM_UNSPECIFIED;
}

static SCM
enqueue_write_buffer_x(SCM s_queue, SCM s_buffer, SCM s_offset, SCM s_size) {
  assert(sizeof(cl_event) == sizeof(scm_t_bits));
  scm_assert_smob_type(cl_command_queue_tag, s_queue);
  scm_assert_smob_type(cl_buffer_tag, s_buffer);
  cl_command_queue queue = (cl_command_queue) SCM_SMOB_DATA(s_queue);
  cl_mem buffer = (cl_mem) SCM_SMOB_DATA(s_buffer);
  int offset = SCM_UNBNDP(s_offset) ? 0 : scm_to_int(s_offset);
  int size = SCM_UNBNDP(s_size)
    ? ((int) SCM_SMOB_DATA_2(s_buffer))
    : scm_to_int(s_size);
  cl_event event;
  cl_int result = clEnqueueWriteBuffer(queue, buffer, CL_FALSE, offset, size,
				       (void *) SCM_SMOB_DATA_3(s_buffer),
				       0, (const cl_event *) NULL,
				       &event);
  if(result != CL_SUCCESS) {
    WARN("Failed to enqueue write buffer %x on queue %x: 0x%x",
	 buffer, queue, result);
    return SCM_BOOL_F;
  }

  return scm_new_smob(cl_event_tag, (scm_t_bits) event);
}

static SCM
enqueue_read_buffer_x(SCM s_queue, SCM s_buffer, SCM s_offset, SCM s_size) {
  assert(sizeof(cl_event) == sizeof(scm_t_bits));
  scm_assert_smob_type(cl_command_queue_tag, s_queue);
  scm_assert_smob_type(cl_buffer_tag, s_buffer);
  cl_command_queue queue = (cl_command_queue) SCM_SMOB_DATA(s_queue);
  cl_mem buffer = (cl_mem) SCM_SMOB_DATA(s_buffer);
  int offset = SCM_UNBNDP(s_offset) ? 0 : scm_to_int(s_offset);
  int size = SCM_UNBNDP(s_size)
    ? ((int) SCM_SMOB_DATA_2(s_buffer))
    : scm_to_int(s_size);
  cl_event event;
  cl_int result = clEnqueueReadBuffer(queue, buffer, CL_FALSE, offset, size,
				      (void *) SCM_SMOB_DATA_3(s_buffer),
				      0, (const cl_event *) NULL,
				      &event);
  if(result != CL_SUCCESS) {
    WARN("Failed to enqueue read buffer %x on queue %x: 0x%x",
	 buffer, queue, result);
    return SCM_BOOL_F;
  }
  return scm_new_smob(cl_event_tag, (scm_t_bits) event);
}


static SCM
enqueue_kernel_x(SCM s_queue, SCM s_kernel, SCM s_dims, SCM s_local_dims) {
  scm_assert_smob_type(cl_command_queue_tag, s_queue);
  scm_assert_smob_type(cl_kernel_tag, s_kernel);
  size_t global_work_size[3];
  cl_uint dims;
  if(scm_is_pair(s_dims)) {
    dims = (cl_uint) scm_to_int(scm_length(s_dims));
    assert(0 < dims && dims <= 3);
    for(int i = 0; scm_is_pair(s_dims); ++i, s_dims = scm_cdr(s_dims)) {
      global_work_size[i] = scm_to_size_t(scm_car(s_dims));
    }
  }
  else {
    dims = 1;
    global_work_size[0] = scm_to_size_t(s_dims);
  }

  size_t *local_work_size = NULL;
  
  if(!SCM_UNBNDP(s_local_dims) ) {
    local_work_size = alloca(dims*sizeof(size_t));
    if(scm_is_pair(s_local_dims)) {
      assert(dims == scm_to_int(scm_length(s_local_dims)));
      for(int i = 0;
	  scm_is_pair(s_local_dims);
	  ++i, s_dims = scm_cdr(s_local_dims)) {
	local_work_size[i] = scm_to_size_t(scm_car(s_local_dims));
	assert(local_work_size[i] <= global_work_size[i]);
      }
    }
    else {
      local_work_size[0] = scm_to_size_t(s_local_dims);
      assert(local_work_size[0] <= global_work_size[0]);
    }
  }
  
  cl_command_queue queue = (cl_command_queue) SCM_SMOB_DATA(s_queue);
  cl_kernel kernel = (cl_kernel) SCM_SMOB_DATA(s_kernel);
  char *kernel_name = (char *) SCM_SMOB_DATA_2(s_kernel);
  
  cl_event event;
  cl_int result = clEnqueueNDRangeKernel(queue, kernel, dims,
					 NULL, global_work_size, local_work_size,
					 0, (const cl_event *) NULL,
					 &event);
  if(result != CL_SUCCESS) {
    WARN("Failed to enqueue kernel %s on queue %x: 0x%x",
	 kernel_name, queue, result);
    return SCM_BOOL_F;
  }

  return scm_new_smob(cl_event_tag, (scm_t_bits) event);
}

static SCM
finish_queue_x(SCM s_queue) {
  scm_assert_smob_type(cl_command_queue_tag, s_queue);
  clFinish((cl_command_queue) SCM_SMOB_DATA(s_queue));
  return SCM_UNSPECIFIED;
}

__declspec(dllexport) void
init() {
  cl_platform_tag = scm_make_smob_type("OpenCL platform", 0);
  cl_device_tag = scm_make_smob_type("OpenCL device", 0);
  cl_context_tag = scm_make_smob_type("OpenCL context", 0);
  cl_command_queue_tag = scm_make_smob_type("OpenCL command queue", 0);
  cl_program_tag = scm_make_smob_type("OpenCL program", 0);
  cl_kernel_tag = scm_make_smob_type("OpenCL kernel", 0);
  cl_buffer_tag = scm_make_smob_type("OpenCL buffer", 0);
  cl_sampler_tag = scm_make_smob_type("OpenCL sampler", 0);
  cl_image2d_tag = scm_make_smob_type("OpenCL 2D image", 0);
  cl_image3d_tag = scm_make_smob_type("OpenCL 3D image", 0);
  cl_event_tag = scm_make_smob_type("OpenCL event", 0);
  
  scm_set_smob_print(cl_platform_tag, platform_smob_print);
  scm_set_smob_print(cl_device_tag, device_smob_print);

  scm_set_smob_print(cl_kernel_tag, kernel_smob_print);
  scm_set_smob_free(cl_kernel_tag, kernel_smob_free);
  
  scm_c_define_gsubr("cl-platforms", 0, 0, 0, platforms);
  scm_c_define_gsubr("cl-devices", 1, 0, 1, devices);
  scm_c_define_gsubr("cl-make-context", 0, 0, 1, create_context);
  scm_c_define_gsubr("call-with-cl-context", 2, 0, 0,
		     call_with_context);
  scm_c_define_gsubr("set-current-cl-context!", 1, 0, 0,
		     set_current_context_x);

  scm_c_define_gsubr("cl-make-command-queue", 1, 0, 1,
		     create_command_queue);
  scm_c_define_gsubr("cl-make-program", 1, 0, 1, create_program);
  scm_c_define_gsubr("call-with-cl-build-options", 2, 0, 0,
		     call_with_build_options);
  scm_c_define_gsubr("cl-kernel", 2, 0, 0, kernel);
  scm_c_define_gsubr("cl-make-buffer", 1, 0, 1, create_buffer);
  scm_c_define_gsubr("cl-bind-arguments", 1, 0, 1, bind_arguments);
  scm_c_define_gsubr("cl-enqueue-read-buffer!", 2, 2, 0, enqueue_read_buffer_x);
  scm_c_define_gsubr("cl-enqueue-write-buffer!", 2, 2, 0, enqueue_write_buffer_x);
  scm_c_define_gsubr("cl-enqueue-kernel!", 3, 1, 0, enqueue_kernel_x);

  scm_c_define_gsubr("cl-finish!", 1, 0, 0, finish_queue_x);
}
