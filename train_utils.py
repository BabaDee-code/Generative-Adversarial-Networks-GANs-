import tensorflow as tf

def generator_loss(disc_generated_output, gen_output, target):
    """Calculates generator loss."""
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (100 * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    """Calculates discriminator loss."""
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

def train_step(input_image, target, generator, discriminator, generator_optimizer, discriminator_optimizer):
    """Executes one training step."""
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        gen_total_loss, _, _ = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
    return gen_total_loss, disc_loss

def fit(train_dataset, generator, discriminator, generator_optimizer, discriminator_optimizer, epochs):
    """Trains the model for the given number of epochs."""
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for input_image, target in train_dataset:
            gen_loss, disc_loss = train_step(input_image, target, generator, discriminator, generator_optimizer, discriminator_optimizer)
        print(f"Generator loss: {gen_loss.numpy():.4f}, Discriminator loss: {disc_loss.numpy():.4f}")
